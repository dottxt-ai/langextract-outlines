import os
import textwrap
from typing import Literal, Union

import langextract as lx
import outlines
import torch
import transformers
from pydantic import BaseModel, Field

from langextract_outlines.provider import OutlinesStructuredGenerationModel


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

# Define the output type
class Character(BaseModel):
    emotional_state: str = Field(description="The emotional state of the character")

class Emotion(BaseModel):
    feeling: str = Field(description="The feeling of the emotion")

class Relationship(BaseModel):
    type: str = Field(description="The type of relationship")

output_type = [Character, Emotion, Relationship]


# 4. Call the extraction function
model_id = "microsoft/Phi-3-mini-4k-instruct"
tf_tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)
tf_model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
)

result = lx.extract(
    text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    prompt_description=prompt,
    examples=examples,
    model=OutlinesStructuredGenerationModel(
        provider_model=outlines.from_transformers(tf_model, tf_tokenizer),
        output_type=output_type,
    ),
)

print(f"Extracted {len(result.extractions)} entities from {len(result.text):,} characters")

# Save and visualize the results
lx.io.save_annotated_documents(
    [result], output_name="romeo_juliet_extractions.jsonl", output_dir="."
)
