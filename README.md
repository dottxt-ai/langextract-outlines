# LangExtract Outlines Plugin

A [LangExtract](https://github.com/google/langextract) provider plugin that integrates [Outlines](https://github.com/outlines-dev/outlines) for structured text extraction using constrained generation.

## Overview

This plugin enables you to use Outlines models with LangExtract for structured information extraction tasks. Outlines provides constrained generation capabilities that ensure model outputs conform to specific schemas, making it ideal for reliable structured extraction.

## Installation

We recommend you use `uv` to install the package.

```bash
uv add langextract-outlines
```

The command above will automatically install langextract and outlines as they are dependencies of langextract-outlines. However, it will not install the optional dependencies required to run specific models with Outlines. If you want to use the Transformers model in Outlines for instance, install the associated optional dependencies:

```bash
uv add outlines[transformers]
```

## Quick Start

To use the `langextract-outlines` plugin, you must provide `OutlinesProvider` instance as the value of the model parameter when using the `langextract.extract` function. As we are directly providing a model, no need to specify a `model_id`.

The arguments to initialize an `OutlinesProvider` instance are very similar to those you would use with the `outlines.Generator` constructor:

- `outlines_model`: an instance of an `outlines.models.Model`, for instance `Transformers` or `MLXLM`
- `output_type`: a list of Pydantic models that will be used to constrain the generation. More information on that in a dedicated section below
- `backend`: the name of the backend that will be used in Outlines to constrain the generation (`outlines_core` by default)
- **`inference_kwargs`: the keyword arguments that will be passed on to the underlying model by Outlines. Those correspond to the argument you would provide when calling a model in Outlines

For instance:

```python
import langextract as lx
import outlines
import transformers
from pydantic import BaseModel, Field
from langextract_outlines import OutlinesProvider

# Define your extraction prompt and examples
prompt = "Extract characters and emotions from the text."
examples = [
    lx.data.ExampleData(
        "Romeo gazed longingly at Juliet.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="Romeo",
                attributes={"emotional_state": "longing"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="longingly",
                attributes={"feeling": "desire"}
            )
        ]
    )
]

# Define the associated output_type
class Character(BaseModel):
    emotional_state: str = Field(description="The emotional state of the character")

class Emotion(BaseModel):
    feeling: str = Field(description="The feeling of the emotion")

output_type = [Character, Emotion]

# Create the Outline model
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

# Create the Outlines provider
outlines_provider = OutlinesProvider(
    outlines_model=outlines.from_transformers(model, tokenizer),
    output_type=output_type,
    backend="outlines_core",
    temperature=0.5,
    repetition_penalty=1,
    max_new_tokens=100,
)

# Run extraction
result = lx.extract(
    "Juliet smiled brightly at the stars.",
    prompt_description=prompt,
    examples=examples,
    model=outlines_provider,
)

print(f"Extracted {len(result.extractions)} entities")
```

## Output Type

The output type you provide must be compatible with the examples as the latter will be included in the prompt. In case of mismatch between the two, generation quality may be severely degraded.

The output type must be a list of Pydantic models, each of them corresponding to an `Extraction` type included in your examples. The name of the Pydantic model must be the name of the `extraction_class` in PascalCase. The fields of the model must correspond to the `attributes` of the extraction instance.

For instance:

```python
import langextract as lx
from pydantic import BaseModel, Field

# Extraction included in the examples
lx.data.Extraction(
    extraction_class="character",
    extraction_text="Romeo",
    attributes={"emotional_state": "longing", "intensity": "medium"}
)

# Possible associated model included in the output_type
class Character(BaseModel):
    emotional_state: str = Field(
        description="The emotional state of the character",
        min_length=1,
        max_length=100,
    )
    intensity: Literal["low", "medium", "high"] = Field(
        description="The intensity of the emotion",
        default="medium",
    )
```

## Inference Arguments

As explained above, all inference arguments such as `temperature`, `max_new_tokens`... must be provided as keyword arguments when initializing the `OutlinesProvider`. Inference arguments specified through other parts of the LangExtract interface will be ignored. Outlines does not standardize inference arguments across models, so you must make sure that the arguments you provide actually correspond to what the model you chose accepts.

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
