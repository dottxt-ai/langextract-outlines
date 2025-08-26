# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Outlines provider for LangExtract."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Union

from langextract import exceptions, inference, schema
from langextract.providers import registry
from outlines.generator import Generator
from pydantic import BaseModel

from .schema import OutlinesSchema

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from outlines.generator import AsyncBlackBoxGenerator, BlackBoxGenerator, SteerableGenerator
    from outlines.models.base import AsyncModel, Model

    # TODO: support other typings?


def _create_extraction_wrapper(user_output_types: list[type[BaseModel]]) -> type[BaseModel]:
    """Turn the user's list of models into the output type expected by LE.
    
    The user should provide a list of Pydantic models. Each model's name is the
    name of an extraction class. The model's fields are the attributes of the
    extraction class.
    """

    from langextract.schema import EXTRACTIONS_KEY
    from pydantic import create_model

    base_models = []

    for user_output_type in user_output_types:
        class_name = user_output_type.__name__
        model = create_model(class_name, **{
            str(class_name.lower()): str,
            f"{class_name.lower()}_attributes": user_output_type
        })
        base_models.append(model)

    return create_model("ExtractionSchema", **{
        EXTRACTIONS_KEY: (list[Union[tuple(base_models)]], ...)
    })


class OutlinesNotInstalledError(exceptions.InferenceConfigError):
    """Exception raised when Outlines is not available.

    This error is raised when trying to use an Outlines provider but the
    optional Outlines dependencies are not installed.
    """


class OutlinesProviderNotInstalledError(exceptions.InferenceConfigError):
    """Exception raised when an Outlines provider is not available.

    This error is raised when trying to use an Outlines provider but the
    optional Outlines provider dependencies are not installed.
    """


@registry.register(
    r"^outlines",
    priority=20,
)
@dataclasses.dataclass(init=False)
class OutlinesStructuredGenerationModel(inference.BaseLanguageModel):
    """
    Language model inference using Outlines for structured generation.
    """

    provider_model: Model | AsyncModel
    output_type: type[BaseModel]
    temperature: float

    _extra_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )
    _generator: SteerableGenerator | BlackBoxGenerator | AsyncBlackBoxGenerator = dataclasses.field(
        init=False, repr=False, compare=False
    )

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
        super().__init__(schema.Constraint(constraint_type=schema.ConstraintType.NONE))

        # TODO: handle provider model kwargs
        # TODO: what other kwargs do we need to optimize the model / generator?
        self.provider_model = provider_model
        self.output_type = output_type
        self.temperature = temperature

        self._extra_kwargs = kwargs or {}
        self._generator = self._create_generator()

        self.set_fence_output(fence_output)


    # TODO: override requires_fence_output to behave sanely

    def apply_schema(self, schema_instance: schema.BaseSchema | None) -> None:
        # ASK: what is this used for if provider kwargs already receives the schema.to_provider_config?
        pass

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
        # TODO: need to fix this error, hard to tell because every underlying provider is different
        # ValueError: The following `model_kwargs` are not used by the model: ['max_workers'] (note: typos in the generate arguments will also show up in this list)
        # merged_kwargs = self.merge_kwargs(kwargs)
        merged_kwargs = {}

        # ASK: should we be using outlines batch or leave that to LE?
        # batch_results = self._generator.batch(list(batch_prompts), **merged_kwargs)
        # scored_results = [
        #     inference.ScoredOutput(score=1.0, output=str(batch_result))
        #     for batch_result in batch_results
        # ]

        # yield scored_results

        try:
            # Process each prompt
            for prompt in batch_prompts:
                try:
                    # Generate using the Outlines generator
                    output = self._generator(
                        prompt,
                        # manually setting these for now until kwargs can be sorted properly
                        temperature=self.temperature,  # tried 0, and increments to 1
                        max_new_tokens=1024,  # tried default, 512 and 1024
                    )

                    # ASK: what is the ScoredOutput for? how should we compute score?
                    # ASK: should outlines perform validation here and throw on fail, or use that for scoring?
                    result = inference.ScoredOutput(score=1.0, output=str(output))
                    yield [result]

                except Exception as e:
                    raise exceptions.InferenceRuntimeError(
                        f"Outlines generation error: {str(e)}",
                        original=e,
                        provider="outlines",
                    ) from e

        except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f"Outlines setup error: {str(e)}", original=e, provider="outlines"
            ) from e

    @classmethod
    def get_schema_class(cls) -> type[OutlinesSchema] | None:
        """Return the OutlinesSchema class for structured output support.

        Returns:
          The OutlinesSchema class that supports strict schema constraints.
        """
        return OutlinesSchema

    def _create_generator(self):
        """Create an Outlines generator using the configured provider and output type."""

        wrapped_output_type = _create_extraction_wrapper(self.output_type)

        return Generator(self.provider_model, wrapped_output_type)
