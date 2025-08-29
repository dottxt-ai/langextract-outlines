import pytest
import textwrap
from typing import Literal

import langextract as lx
from pydantic import BaseModel, Field
from outlines import from_transformers
from outlines.types.dsl import is_pydantic_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from langextract_outlines.provider import OutlinesProvider, _format_output_type


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


class Character(BaseModel):
    emotional_state: str = Field(
        description="The emotional state of the character"
    )


class Emotion(BaseModel):
    feeling: str = Field(
        description="The feeling of the emotion",
        min_length=1,
        max_length=100,
    )
    intensity: Literal["low", "medium", "high"] = Field(
        description="The intensity of the emotion",
        default="medium",
    )


def test_format_output_type():
    # Single class
    output_types = [Character]
    formatted_output_type = _format_output_type(output_types)

    assert is_pydantic_model(formatted_output_type)
    extractions = formatted_output_type.model_fields["extractions"].annotation
    character = extractions.__args__[0]
    assert character.model_fields["character"].annotation == str
    assert character.model_fields["character_attributes"].annotation == Character

    # Multiple classes
    output_types = [Character, Emotion]
    formatted_output_type = _format_output_type(output_types)

    assert is_pydantic_model(formatted_output_type)
    extractions = formatted_output_type.model_fields["extractions"].annotation
    union = extractions.__args__[0]
    character = union.__args__[0]
    assert character.model_fields["character"].annotation == str
    assert character.model_fields["character_attributes"].annotation == Character
    emotion = union.__args__[1]
    assert emotion.model_fields["emotion"].annotation == str
    assert emotion.model_fields["emotion_attributes"].annotation == Emotion


def test_outlines_provider():
    prompt = textwrap.dedent("""\
        Extract characters, emotions, and relationships in order of appearance.
        Use exact text for extractions. Do not paraphrase or overlap entities.
        Provide meaningful attributes for each entity to add context. Extract at most 3 entities.""")

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
                    attributes={"feeling": "gentle awe", "intensity": "high"},
                ),
            ],
        )
    ]

    outlines_model = from_transformers(
        AutoModelForCausalLM.from_pretrained(MODEL_NAME),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    provider = OutlinesProvider(
        outlines_model,
        [Character, Emotion],
        backend="outlines_core",
        max_new_tokens=500,
    )

    result = lx.extract(
        text_or_documents="Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
        prompt_description=prompt,
        examples=examples,
        model=provider,
    )

    assert result is not None
    assert isinstance(result.extractions, list)
    for extraction in result.extractions:
        assert extraction.extraction_class in ["character", "emotion"]
        assert extraction.extraction_text is not None
        assert extraction.attributes is not None
        assert isinstance(extraction.attributes, dict)
        attributes_keys = extraction.attributes.keys()
        if extraction.extraction_class == "character":
            assert all(key in ["emotional_state"] for key in attributes_keys)
        elif extraction.extraction_class == "emotion":
            assert all(key in ["feeling", "intensity"] for key in attributes_keys)
