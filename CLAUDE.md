# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Paper: "Simple Agents Outperform Experts in Biomedical Imaging Workflow Optimization" (Wang, Horstmann, Lin, Chen, Farhang, Stiles, Sehgal, Light, Van Valen, Yue, Sun — Caltech, Cornell, UT Austin, RPI). [arXiv:2512.06006](https://arxiv.org/abs/2512.06006)

AI agents (AutoGen + OpenAI/Together) autonomously write and iteratively optimize preprocessing/postprocessing functions for scientific image analysis tasks. A function bank accumulates all generated solutions with metrics, feeding the best/worst back into prompts to guide exploration.

### Three Tasks and Metrics

| Task | Tool | Metric | Expert Baseline |
|------|------|--------|-----------------|
| Molecular (spot detection) | Polaris/DeepCell | F1 score | 0.841 |
| Cellular (cell segmentation) | Cellpose 3 (cyto3) | AP @ IoU 0.5 | 0.402 |
| Macroscopic (medical segmentation) | MedSAM | NSD + DSC | 0.820 |

### Experiment Configuration

- LLMs tested: GPT-4.1, o3, Llama 3.3-70B-Instruct-Turbo
- Default: 20 iterations x k=3 function pairs = 60 trials per run, 20 runs per config
- Function bank sampling: top 3 and bottom 3 performing functions fed back into prompts
- AutoML (Optuna) search: every 5 iterations, 24 trials per function on top-3 functions
- API library: 98 curated functions from OpenCV, scikit-image, SciPy (`assets/APIs.txt`)

## Environment Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[cellpose]"   # or .[polaris], .[medsam], .[dev]
export OPENAI_API_KEY="..."
# Task-specific: DEEPCELL_ACCESS_TOKEN for Polaris, checkpoint download for MedSAM
```

## Running Experiments

```bash
python main.py \
    --dataset $DATA_PATH --gpu_id 0 \
    --experiment_name cellpose_segmentation \
    --random_seed 42 -k 3 \
    --history_threshold 5
```

Key flags: `--hyper_optimize` (Optuna refinement post-trial), `--checkpoint_path` (MedSAM only), `--llm_model` (default: gpt-4.1), `--max_round` (default: 20), `--cache_seed` (default: 4).

Results go to `{experiment_name}/{timestamp}/preprocessing_func_bank.json`.

## Running Tests

Unit tests (no GPU/data required):
```bash
python -m pytest tests/ -v  # runs all unit tests (integration tests are marked and skipped by default)
```

Integration tests (require data paths and task-specific packages):
```bash
python -m tests.test_cellpose_segmentation --data_path /path/to/data
python -m tests.test_spotdetection
python -m tests.test_medsam_segmentation
```

## Architecture

### Core Loop (`main.py`)

Each iteration: construct prompt (task details + function bank sample) → code_writer_agent generates k preprocessing/postprocessing function pairs → TemplatedLocalCommandLineCodeExecutor injects code into execution template at `# --- CODEGEN_PREPROCESSING_FUNCTIONS_INSERT ---` placeholder → execute and extract metrics → store results in JSON function bank → repeat with updated context.

Two AutoGen agents alternate via `state_transition`: **code_writer_agent** (LLM-powered, generates code) and **code_executor_agent** (runs code, returns output). The writer must produce exactly k function pairs named `preprocess_images_i`/`postprocess_preds_i` per iteration.

### Key Abstractions

- **`ImageData`** (`src/data_io.py`): Framework-agnostic batched image container. Images are H,W,C format. Supports variable resolutions within a batch (stored as list of arrays). Also has `ImageDataNP` for uniform-size numpy arrays.

- **`TaskPrompts`** (`prompts/task_prompts.py`): Dataclass base class. Concrete `run_pipeline_prompt()` reads `prompts/execution_template.py.txt` and applies task-specific + standard replacements. Each task subclass implements `get_template_replacements()`, `get_task_details()`, `get_pipeline_metrics_info()`. The placeholder constant is `_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER`.

- **`TemplatedLocalCommandLineCodeExecutor`** (`utils/executors.py`): Extends AutoGen's `LocalCommandLineCodeExecutor`. Takes a template script callable and placeholder string. Injects agent-generated code into the template, also embeds the code as `_GENERATED_CODE_STRING` variable for logging.

- **Function bank** (`utils/function_bank_utils.py`): JSON array where each entry has `preprocessing_function`, `postprocessing_function`, `overall_metrics`, `expression`. Utilities: `top_n()`, `worst_n()`, `last_n()` with custom sorting functions.

- **Hyperparameter optimization** (`hyper_optimize.py`): Post-trial refinement. Extracts OpenCV parameter constants from generated code via AST, creates Optuna search space respecting constraints from `assets/opencv_arg_rules.py`, optimizes parameters.

- **`analysis_utils`** (`utils/analysis_utils.py`): Shared utilities for analysis scripts — `find_all_metrics()`, `find_highest()`, `find_rolling_highest()`, `find_top_k()`, `convert_string_to_function()`, `parse_automl_status()`.

- **Task registry** (`main.py:TASK_CONFIGS`): Dict mapping experiment names to prompt class, sampling function, and extra kwargs. Adding a new task only requires a new dict entry.

### Adding a New Task

Three components required:
1. **Tool wrapper** in `src/{task}.py` — class with `__init__()`, `predict()`, `evaluate()` methods
2. **Prompts** in `prompts/{task}_prompts.py` — inherits `TaskPrompts`, implements `get_template_replacements()` (task-specific template placeholders), `get_task_details()`, `get_pipeline_metrics_info()`
3. **Task registry entry** in `main.py:TASK_CONFIGS` — prompt class, sampling function, extra kwargs
4. **Expert baseline** in `prompts/{task}_expert_preprocessing.py.txt` and `prompts/{task}_expert_postprocessing.py.txt`

The shared execution template (`prompts/execution_template.py.txt`) uses task-specific placeholders filled by `get_template_replacements()`. No per-task template file needed.

### Important Constraints

- Generated code must be **stateless and self-contained**: all imports inside the function, no reliance on external state between iterations
- Functions must not define inner/helper functions — all logic goes inside the main function body
- The execution template placeholder string must match exactly: `# --- CODEGEN_PREPROCESSING_FUNCTIONS_INSERT ---`
- Cellpose task: may need to comment out `fill_holes_and_remove_small_masks` in cellpose `dynamics.resize_and_compute_masks`

## Analysis Scripts

```bash
# Analyze trajectories (creates analysis_results/ under each result folder)
python figs/cellpose_analyze_trajectories.py --data_path=$DATA
python figs/spot_detection_analyze_trajectories.py --checkpoint_path=... --val_data_path=... --test_data_path=... --gpu_id=0
python figs/medsam_analyze_trajectories.py --data_path=$DATA --gpu_id=0

# Aggregate across repetitions
python aggregate_results_across_reps.py --task_name cellpose_segmentation
```
