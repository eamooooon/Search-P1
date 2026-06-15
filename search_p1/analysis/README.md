# Search-P1 Analysis Utilities

This directory contains offline utilities for building, checking, and debugging
reference search paths from rollout dumps.

## Current Reference Path Pipeline

- `build_reference_steps_llm_direct.py`: build final reference path rows by
  calling an OpenAI-compatible LLM endpoint directly. Used by
  `scripts/hotpotqa_p1/build_reference_steps_llm.sh`.
- `reference_llm.py`: shared LLM request, response parsing, retry, and
  reference-step validation helpers.
- `reference_voting.py`: builds prompts from successful trajectories and
  converts consensus/LLM outputs into reference steps.
- `reference_sampling.py`: groups successful trajectories and extracts valid
  search actions.
- `reference_io.py`: shared JSONL/key/question helpers.
- `reward_format.py`: local import shim for the reward parser.

## Rollout Analysis And Data Export

- `analyze_reference_rollouts.py`: summarize rollout correctness and coverage.
- `export_corrected_reference_rollouts.py`: export corrected/usable successful
  trajectories for reference construction.
- `export_resample_questions.py`: export questions whose correct trajectory
  count is below a threshold for targeted resampling.
