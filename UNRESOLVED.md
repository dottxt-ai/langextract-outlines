## langextract team

- [langextract team] confusing behavior around `fence_outputs`
  - is this resolvable on LE side or is this plugin just fighting against the grain? would be nice to not have to declare it in both model and extract / resolve params

## outlines team

- strategy for data modeling the output type is unclear
  - adherence to the `{ EXTRACTIONS_KEY: OutputType[] }` structure
    - using the provider wrapper util is working but may not be optimal (mypy/pylance arent happy with it although it works at runtime)
  - how do you reconcile the loose / multi-schema nature proposed by the examples (and therefore the prompt) with the strict single-schema nature of outlines?
  - how would the R/J example be modeled (or should it even be using outlines)?
  - does outlines need its own examples that are more fitting for its use case?
- temperature param is being ignored when passed to the generator 
  - `The following generation flags are not valid and may be ignored: ['temperature']`
- how to normalize generator (provider model) kwargs so they can be passed through agnostically?
  - transformers (and maybe others) will throw if they encounter an unexpected kwarg
  - not all provider models have the same kwargs and may also throw on unexpected ones

## cross-team

- should an alternative approach be considered instead of the hard override extract model param route?
  - it feels a bit hacky / escape-hatchy relative to the direction LE is suggesting for custom providers
  - maybe thats the nature of this type of provider which is more of a model provider wrapper / middleware than LE other custom provider examples (gemini, openai etc)
    - maybe its worth considering a new type of `BaseStructuredGenerationProvider`? 
- results are producing `.jsonl` but are actually single JSON objects with a top level `extractions` list property
  - what could be the cause of this?
  - what are the implications?
    - while it doesnt throw this does break the expected API of LE and at the very least needs to be documented 
- (possibly related to the above) the alignment status varied significantly both within a single extraction and across the different output types
- outlines supports (provider model dependent) batch results but LE expects a single scored result to be yielded
  - technically what is yielded is a list with one scored result
  - both the Gemini and OpanAI providers implement a batch approach that is (nearly or totally) identical
  - should a `infer_batch` or something be standardized so that standard model providers can just implement `infer` (single inference) and structured generation providers could override `infer_batch` with their own implementation?