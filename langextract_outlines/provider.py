"""Outlines provider for LangExtract."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Optional, Union

from langextract import exceptions, inference, schema
from langextract.providers import registry
from langextract.schema import EXTRACTIONS_KEY
from outlines.generator import Generator
from outlines.models.base import AsyncModel, Model
from outlines.types.dsl import is_pydantic_model
from pydantic import BaseModel, create_model

OutputTypes = Sequence[type[BaseModel]]


@registry.register(
    r"^outlines",
    priority=20,
)
class OutlinesProvider(inference.BaseLanguageModel):
    """Custom provider to use Outlines with LangExtract."""

    def __init__(
        self,
        outlines_model: Model | AsyncModel,
        output_types: Optional[OutputTypes] = None,
        backend: Optional[str] = None,
        **inference_kwargs,
    ) -> None:
        """
        Parameters
        ----------
        outlines_model:
            The Outlines Model instance to use for inference
        output_types:
            A list of Pydantic models that correspond to the extraction classes
            used in the examples. The value provided will be turned into a
            Pydantic model that that will be used by Outlines to constrain the
            output.
        backend:
            The backend Outlines will use to constrain the generation.
        **inference_kwargs:
            Inference parameters passed down to the `__call__` method of the
            Outlines generator. For instance, `temperature` or `max_tokens`.
        """
        super().__init__(schema.Constraint(constraint_type=schema.ConstraintType.NONE))

        formatted_output_types = _format_output_type(output_types)
        self._generator = Generator(
            outlines_model,
            formatted_output_types,
            backend,
        )
        self._inference_kwargs = inference_kwargs or {}
        self.set_fence_output(False)

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[inference.ScoredOutput]]:
        """Runs inference on a list of prompts via Outlines.

        Args:
          batch_prompts: A list of string prompts.
          **kwargs: Additional generation params (temperature, max_tokens, etc.)

        Yields:
          Lists of ScoredOutputs.
        """
        try:
            # We try to use the batch method, but if it's not supported,
            # we fall back to iterating over the prompts and calling the
            # __call__ method
            try:
                output = self._generator.batch(batch_prompts, **self._inference_kwargs)
            except NotImplementedError:
                output = []
                for prompt in batch_prompts:
                    output.append(self._generator(prompt, **self._inference_kwargs))
            yield [
                inference.ScoredOutput(score=1.0, output=str(output))
                for output in output
            ]
        except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f"Outlines setup error: {str(e)}", original=e, provider="outlines"
            ) from e


def _format_output_type(
    user_output_types: Optional[OutputTypes] = None,
) -> Optional[type[BaseModel]]:
    """Turn the user's list of pydantic models into the output type
    expected by LE.

    The user should provide a list of Pydantic models. Each model's name is the
    name of an extraction class. The model's fields are the attributes of the
    extraction class.

    Parameters
    ----------
    user_output_types:
        A list of Pydantic models that correspond to the extraction classes
        used in the examples.

    Returns
    -------
    Optional[type[BaseModel]]
        A Pydantic model that will be used by Outlines to constrain the
        output. The model will have a field called `extractions` that will
        be a list of the user's Pydantic models corresponding to the
        extraction classes.

    """
    if not user_output_types:
        return None

    if not isinstance(user_output_types, list) or not all(
        is_pydantic_model(item) for item in user_output_types
    ):
        raise ValueError(
            "The `output_types` parameter must be a list of Pydantic "
            "models. Got: " + str(user_output_types)
        )

    base_models = []

    for user_output_type in user_output_types:
        # Each extraction class must have a name and attributes
        # in a field called `<name>_attributes`
        class_name = user_output_type.__name__
        model = create_model(
            class_name,
            **{
                str(class_name.lower()): str,
                f"{class_name.lower()}_attributes": user_output_type,
            },
        )
        base_models.append(model)

    return create_model(
        "ExtractionSchema",
        **{EXTRACTIONS_KEY: (list[Union[tuple(base_models)]], ...)},  # type: ignore
    )
