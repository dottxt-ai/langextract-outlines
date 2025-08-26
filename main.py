import os
import textwrap
from typing import Literal

import langextract as lx
import outlines
import torch
import transformers
from pydantic import BaseModel, Field

from langextract_outlines.provider import OutlinesStructuredGenerationModel

# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
# huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
#         - Avoid using `tokenizers` before the fork if possible
#         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"},
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"},
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"},
            ),
        ],
    )
]


# ASK: not sure how to model this
# see RESULTS.md 

# OPTION 1
class ExtractionSchema(BaseModel):
    extraction_text: str
    extraction_class: Literal["character"] | Literal["emotion"] | Literal["relationship"]


# class ExtractionBase(BaseModel):
#     extraction_text: str


# class Character(ExtractionBase):
#     extraction_class: Literal["character"] = "character"
#     emotional_state: str


# class Emotion(ExtractionBase):
#     extraction_class: Literal["emotion"] = "emotion"
#     feeling: str


# class Relationship(ExtractionBase):
#     extraction_class: Literal["relationship"] = "relationship"
#     type: str

# OPTION 2
# class ExtractionSchema(BaseModel):
#     extraction: Character | Emotion | Relationship = Field(discriminator="extraction_class")

# OPTION 3
# class ExtractionSchema(BaseModel):
#     character: Character | None = None
#     emotion: Emotion | None = None
#     relationship: Relationship | None = None

# OPTION 4
# class ExtractionSchema(BaseModel):
#     extraction_text: str = Field(description="the raw text used for this extraction")
#     extraction_class: Literal["character"] | Literal["emotion"] | Literal["relationship"] = Field(description="classification of the extraction")
#     emotional_state: str | None = Field(description="only used for extraction_class `character`, otherwise None", default=None)
#     feeling: str | None = Field(description="only used for extraction_class `emotion`, otherwise None", default=None)
#     type: str | None = Field(description="only used for extraction_class `relationship`, otherwise None", default=None)


model_id = "microsoft/Phi-3-medium-4k-instruct"
tf_tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)
tf_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="mps",
)

result = lx.extract(
    text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    prompt_description=prompt,
    examples=examples,
    max_workers=1,
    extraction_passes=1,
    # max_workers=20,  # Parallel processing for speed
    max_char_buffer=1000,  # Smaller contexts for better accuracy
    model=OutlinesStructuredGenerationModel(
        provider_model=outlines.from_transformers(tf_model, tf_tokenizer),
        output_type=ExtractionSchema,
        # NOTE: transformers warns that this param is being ignored
        temperature=0.1
    ),
)

print(f"Extracted {len(result.extractions)} entities from {len(result.text):,} characters")

# Save and visualize the results
lx.io.save_annotated_documents(
    [result], output_name="romeo_juliet_extractions.jsonl", output_dir="."
)
