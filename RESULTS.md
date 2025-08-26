# results with different output type modeling strategies

this illustrates the following unresolved challenge:

> how do you reconcile the loose / multi-schema nature proposed by the examples (and therefore the prompt) with the strict single-schema nature of outlines?

## prompts and settings

all used the same prompts, examples and settings

<details>
  <summary>view</summary>

  ```py
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

  ...

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
  ```
</details>


## option 1

```py
class SimpleExtraction(BaseModel):
    extraction_text: str
    extraction_class: Literal["character"] | Literal["emotion"] | Literal["relationship"]
```

<details>
  <summary>JSON result</summary>

  ```json
  {
    "extractions": [
      {
        "extraction_class": "extraction_text",
        "extraction_text": "Juliet",
        "char_interval": {
          "start_pos": 5,
          "end_pos": 11
        },
        "alignment_status": "match_exact",
        "extraction_index": 1,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 2,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_text",
        "extraction_text": "Romeos",
        "char_interval": {
          "start_pos": 63,
          "end_pos": 68
        },
        "alignment_status": "match_fuzzy",
        "extraction_index": 3,
        "group_index": 1,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 4,
        "group_index": 1,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_text",
        "extraction_text": "longingly",
        "char_interval": {
          "start_pos": 18,
          "end_pos": 27
        },
        "alignment_status": "match_exact",
        "extraction_index": 5,
        "group_index": 2,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "emotion",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 6,
        "group_index": 2,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_text",
        "extraction_text": "aching",
        "char_interval": {
          "start_pos": 52,
          "end_pos": 58
        },
        "alignment_status": "match_exact",
        "extraction_index": 7,
        "group_index": 3,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "emotion",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 8,
        "group_index": 3,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 2,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 4,
        "group_index": 1,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "emotion",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 6,
        "group_index": 2,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "emotion",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 8,
        "group_index": 3,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 2,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 4,
        "group_index": 1,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "emotion",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 6,
        "group_index": 2,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "emotion",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 8,
        "group_index": 3,
        "description": null,
        "attributes": null
      }
    ],
    "text": "Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    "document_id": "doc_e8afcd98"
  }
  ```
</details>

## option 2

```py
class ExtractionBase(BaseModel):
    extraction_text: str


class Character(ExtractionBase):
    extraction_class: Literal["character"] = "character"
    emotional_state: str


class Emotion(ExtractionBase):
    extraction_class: Literal["emotion"] = "emotion"
    feeling: str


class Relationship(ExtractionBase):
    extraction_class: Literal["relationship"] = "relationship"
    type: str

class ExtractionSchema(BaseModel):
    extraction: Character | Emotion | Relationship = Field(discriminator="extraction_class")
```

<details>
  <summary>JSON result</summary>

  ```json
  {
    "extractions": [
      {
        "extraction_class": "extraction",
        "extraction_text": "{'extraction_text': 'Juliet', 'extraction_class': 'character', 'emotional_state': 'longing'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 1,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction",
        "extraction_text": "{'extraction_text': 'Romeо', 'extraction_class': 'character', 'emotional_state': 'absent but influential'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 2,
        "group_index": 1,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction",
        "extraction_text": "{'extraction_text': 'stars', 'extraction_class': 'emotion', 'feeling': 'longing'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 3,
        "group_index": 2,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction",
        "extraction_text": "{'extraction_text': 'heart aching', 'extraction_class': 'emotion', 'feeling': 'sorrow'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 4,
        "group_index": 3,
        "description": null,
        "attributes": null
      }
    ],
    "text": "Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    "document_id": "doc_d5963613"
  }
  ```
</details>

## option 3

```py
class ExtractionBase(BaseModel):
    extraction_text: str


class Character(ExtractionBase):
    extraction_class: Literal["character"] = "character"
    emotional_state: str


class Emotion(ExtractionBase):
    extraction_class: Literal["emotion"] = "emotion"
    feeling: str


class Relationship(ExtractionBase):
    extraction_class: Literal["relationship"] = "relationship"
    type: str

class ExtractionSchema(BaseModel):
    character: Character | None = None
    emotion: Emotion | None = None
    relationship: Relationship | None = None
```

<details>
  <summary>JSON result</summary>

  ```json
  {
    "extractions": [
      {
        "extraction_class": "character",
        "extraction_text": "{'extraction_text': 'Juliet', 'extraction_class': 'character', 'emotional_state': 'longing'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 1,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "character",
        "extraction_text": "{'extraction_text': 'Rome', 'extraction_class': 'character', 'emotional_state': 'absent'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 2,
        "group_index": 1,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "emotion",
        "extraction_text": "{'extraction_text': 'longingly', 'extraction_class': 'emotion', 'feeling': 'yearning'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 3,
        "group_index": 2,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "emotion",
        "extraction_text": "{'extraction_text': 'heart aching', 'extraction_class': 'emotion', 'feeling': 'pain'}",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 4,
        "group_index": 3,
        "description": null,
        "attributes": null
      }
    ],
    "text": "Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    "document_id": "doc_5f9414cf"
  }
  ```
</details>

## option 4

i dont think the model gets to see these descriptions. maybe if it were blended into the prompt it could help?

```py
class ExtractionSchema(BaseModel):
    extraction_text: str = Field(description="the raw text used for this extraction")
    extraction_class: Literal["character"] | Literal["emotion"] | Literal["relationship"] = Field(description="classification of the extraction")
    emotional_state: str | None = Field(description="only used for extraction_class `character`, otherwise None", default=None)
    feeling: str | None = Field(description="only used for extraction_class `emotion`, otherwise None", default=None)
    type: str | None = Field(description="only used for extraction_class `relationship`, otherwise None", default=None)
```

<details>
  <summary>JSON result</summary>

  ```json
  {
    "extractions": [
      {
        "extraction_class": "extraction_text",
        "extraction_text": "Juliet",
        "char_interval": {
          "start_pos": 5,
          "end_pos": 11
        },
        "alignment_status": "match_exact",
        "extraction_index": 1,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "extraction_class",
        "extraction_text": "character",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 2,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "emotional_state",
        "extraction_text": "longing",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 3,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "feeling",
        "extraction_text": "heartache",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 4,
        "group_index": 0,
        "description": null,
        "attributes": null
      },
      {
        "extraction_class": "type",
        "extraction_text": "romantic interest",
        "char_interval": null,
        "alignment_status": null,
        "extraction_index": 5,
        "group_index": 0,
        "description": null,
        "attributes": null
      }
    ],
    "text": "Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    "document_id": "doc_30777a31"
  } 
  ```
</details>