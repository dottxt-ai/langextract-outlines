# notes

notes taken along the way to save someone else the hassle. should be deleted before releasing

## the `fence_output` problem

> `fence_output` has multiple use cases and implicit interpretations

we are providing the model directly so in theory we should be overriding all other config and options..

- fixed in the model itself by invoking the base model method to set the choice
- the resolver though ALSO gets its own value for `fence_output` which is distinct from the model leading to `Input string does not contain valid markers`

resolve abstract base defaults to `True`

```py
class AbstractResolver(abc.ABC):
  """Resolves LLM text outputs into structured data."""

  # TODO: Review value and requirements for abstract class.
  def __init__(
      self,
      fence_output: bool = True,
      constraint: schema.Constraint = schema.Constraint(),
      format_type: data.FormatType = data.FormatType.JSON,
  ):
```

concrete resolve does too

```py
class Resolver(AbstractResolver):
  """Resolver for YAML/JSON-based information extraction.

  Allows for customized parsing of YAML or JSON content within text. Extracted
  extractions are either sorted by a specified index suffix, or, if this is not
  present, extractions are ordered by their appearance in the order they appear.
  Attributes associated with extractions are extracted if an attributes suffix
  is
  provided. Both the index and attributes suffixes are dictated by prompt
  examples.
  """

  def __init__(
      self,
      fence_output: bool = True,
```

concrete gets instantied in `extract` 

```py
...
fence_output = language_model.requires_fence_output

resolver_defaults = {
    "fence_output": fence_output,
    "format_type": format_type,
    "extraction_attributes_suffix": "_attributes",
    "extraction_index_suffix": None,
}
resolver_defaults.update(resolver_params or {})
...

...
res = resolver.Resolver(**resolver_defaults)
```

somehow this is not resulting in `False`. the logic above suggests that the model should be the source of truth

```py
# OutlinesStructuredGenerationModel:
def __init__(
    self,
    provider_model: Model | AsyncModel,
    # ASK: provider model and generator accept Any | None but dont we want to type this param?
    output_type: type[BaseModel],
    temperature: float = 0.0,
    fence_output: bool = False,
    **kwargs,
) -> None:
    """Initialize the Outlines structured generation model.

    Args:
        model: Outlines provider model
        output_type: The output type to constrain to
        temperature: Sampling temperature.
        fence_output: Whether to wrap output in markdown fences
        **kwargs: Additional parameters passed to Outlines provider.

    Raises:
      InferenceConfigError: If model ID format is invalid.
    """

    # TODO: handle provider model kwargs
    # TODO: what other kwargs do we need to optimize the model / generator?
    self.provider_model = provider_model
    self.output_type = output_type
    self.temperature = temperature

    self._extra_kwargs = kwargs or {}
    self._generator = self._create_generator()

    self.set_fence_output(fence_output)
    super().__init__(schema.Constraint(constraint_type=schema.ConstraintType.NONE))
    # debugger at this point says 
    # fence_output = False
    # self.requires_fence_output = False
    # self._fence_output_override = False
...
```

in `extract()`

```py
language_model = None

if model:
  language_model = model # OutlinesStructuredGenerationModel
  if fence_output is not None: # fence_output = None (extract param default)
    # skips this line
    language_model.set_fence_output(fence_output)
...
# fence_output = None (extract param default)
fence_output = language_model.requires_fence_output
# BUT language_model.requires_fence_output = True???
```

and this is happening because of this bullshit on the base language model. since this SG provider model doesnt use a schema class it will fail this implementation. then wtf is the point of the `fence_output` param if schema (or lack thereof) fundamentally controls all downstream usage of the `fence_output`...

```py
@property
def requires_fence_output(self) -> bool:
  """Whether this model requires fence output for parsing.

  Uses explicit override if set, otherwise computes from schema.
  Returns True if no schema or schema doesn't support strict mode.
  """
  if (
      hasattr(self, '_fence_output_override')
      and self._fence_output_override is not None
  ):
    return self._fence_output_override
  if not hasattr(self, '_schema') or self._schema is None:
    return True
  return not self._schema.supports_strict_mode
```

in the end the issue was calling `super().__init__` at the end of the outlines class `__init__`. the base `__init__` sets the `_fence_output_override` to `None` which was resetting the work done by `self.set_fence_output(fence_output)`.

i found this strange since every other language i have worked in has required super calls to always be done first, but figured there must be a reason for it since all the other LE provider implementations do it. lesson learned..


