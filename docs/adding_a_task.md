# Adding a New Task

This guide walks through adding a new scientific imaging task to the agent optimization framework. By the end, the framework's LLM agents will iteratively generate and test preprocessing/postprocessing functions for your tool.

## Overview

You need to create four things:

1. **Tool wrapper** (`src/your_task.py`) — wraps your model's predict + evaluate
2. **Prompt subclass** (`prompts/your_task_prompts.py`) — tells the LLM about your task
3. **Registry entry** (in `main.py:TASK_CONFIGS`) — wires everything together
4. **Expert baselines** (`prompts/your_task_expert_*.py.txt`) — reference implementations

## Step 1: Tool Wrapper

Create `src/your_task.py` with a class that implements `predict()` and `evaluate()`.

See `src/base_tool.py` for the protocol definition. Your class doesn't need to inherit from it — just match the method signatures.

```python
from src.data_io import ImageData

class YourTool:
    def __init__(self, device: int = 0, **kwargs):
        # Load your model here
        ...

    def predict(self, images: ImageData, **kwargs):
        # Run inference. Input is preprocessed ImageData.
        # Return predictions in whatever format your evaluate() expects.
        ...

    def evaluate(self, predictions, ground_truth) -> dict:
        # Compare predictions to ground truth.
        # MUST return a dict with string keys and float values.
        return {"your_metric": 0.85}
```

### Critical contract: evaluate() return value

The returned dict is stored verbatim as `"overall_metrics"` in the function bank JSON. Your `sampling_function` in TASK_CONFIGS (step 3) must be able to extract a single scalar from it:

```python
# If evaluate() returns {"my_score": 0.85}
# Then sampling_function must be: lambda x: x['overall_metrics']['my_score']
```

## Step 2: Prompt Subclass

Create `prompts/your_task_prompts.py` inheriting from `TaskPrompts`:

```python
from prompts.task_prompts import TaskPrompts


class YourTaskPrompts(TaskPrompts):
    # Class attribute: describe your dataset for the LLM
    dataset_info = """
    ```markdown
    Description of your dataset: channels, dimensions, value ranges, etc.
    Images have dimensions (B, H, W, C) = (batch, height, width, channels).
    ```
    """

    # Optional: provide a postprocessing skeleton file
    postprocessing_skeleton_filename = "your_task_expert_postprocessing_skeleton.py.txt"

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word,
                 # Add any extra kwargs your task needs:
                 batch_size=16):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            function_bank_path=function_bank_path,
            k=k,
            k_word=k_word,
            batch_size=batch_size,
        )

    def get_template_replacements(self) -> dict:
        """Return task-specific placeholder values for execution_template.py.txt"""
        return {
            "task_extra_imports": "",
            "task_imports": (
                "    from src.your_task import YourTool\n"
                "    from src.utils import set_gpu_device"
            ),
            "task_extra_config": "set_gpu_device(gpu_id)",
            "task_setup": (
                '    tool = YourTool(device=gpu_id)\n'
                '    # Load data and create ImageData objects\n'
                '    raw, gt = tool.loadData(data_path)\n'
                '    images = ImageData(raw=raw, batch_size=len(raw))\n'
                '    single_image_data = ImageData(raw[:2], batch_size=1)'
            ),
            "task_predict_single": "tool.predict(preprocessed_images)",
            "task_postprocess_check_single": "",
            "task_evaluate_single": "        k_overall_metrics_single_img.append(tool.evaluate(pred, gt[:1]))",
            "task_predict_full": "tool.predict(preprocessed_images)",
            "task_postprocess_check_full": "",
            "task_evaluate_full": "        k_overall_metrics.append(tool.evaluate(pred, gt))",
            "task_metrics_log": '    logger.info("All overall metrics: %s", json.dumps(k_overall_metrics if k_overall_metrics else \'N/A\', indent=2))',
        }

    def get_task_details(self):
        return f"""
    Write {self.k_word} pairs of preprocessing and postprocessing functions.
    [Describe what the agent should optimize and any constraints]
    """

    def get_pipeline_metrics_info(self):
        return """
    Metrics: your_metric (higher is better)
    your_metric: [describe what it measures]
    """
```

### Template placeholders reference

Your `get_template_replacements()` must return values for all these keys. They fill slots in `prompts/execution_template.py.txt`:

| Placeholder | Purpose | Example |
|---|---|---|
| `task_extra_imports` | Top-level imports outside the try block | `""` or `"import something"` |
| `task_imports` | Imports inside the try block (indented 4 spaces) | `"    from src.your_task import YourTool"` |
| `task_extra_config` | Config after gpu_id/seed/paths are set | `"set_gpu_device(gpu_id)"` |
| `task_setup` | Load model + data, create `images` and `single_image_data` | See examples |
| `task_predict_single` | Expression calling predict on single image | `"tool.predict(preprocessed_images)"` |
| `task_predict_full` | Expression calling predict on full dataset | `"tool.predict(preprocessed_images)"` |
| `task_postprocess_check_single` | Optional validation after postprocessing (single) | `""` |
| `task_postprocess_check_full` | Optional validation after postprocessing (full) | `""` |
| `task_evaluate_single` | Append single-image metrics (8-space indent) | `"        k_overall_metrics_single_img.append(tool.evaluate(pred, gt[:1]))"` |
| `task_evaluate_full` | Append full metrics (8-space indent) | `"        k_overall_metrics.append(tool.evaluate(pred, gt))"` |
| `task_metrics_log` | Log metrics (4-space indent) | See examples |

**Important**: The `task_setup` replacement must define two variables:
- `images` — full dataset as `ImageData`
- `single_image_data` — 1-2 images as `ImageData` (used for fast-fail testing)

### Standard placeholders (handled automatically)

These are filled by `TaskPrompts._get_standard_replacements()` — you don't need to provide them:

`{gpu_id}`, `{seed}`, `{dataset_path}`, `{function_bank_path}`, `{sample_k}`, `{checkpoint_path}`, `{dataset_size}`, `{batch_size}`

## Step 3: Register in TASK_CONFIGS

In `main.py`, add your import and registry entry:

```python
from prompts.your_task_prompts import YourTaskPrompts

TASK_CONFIGS = {
    # ... existing tasks ...
    "your_task": {
        "prompt_class": YourTaskPrompts,
        "sampling_function": lambda x: x['overall_metrics']['your_metric'],
        "sampling_function_source": "sampling_function = lambda x: x['overall_metrics']['your_metric']",
        "extra_kwargs": {"batch_size": 16},  # passed to prompt_class.__init__
    },
}
```

### Fields explained

| Field | Type | Purpose |
|---|---|---|
| `prompt_class` | class | Your TaskPrompts subclass |
| `sampling_function` | callable | Extracts a scalar from a function bank entry for ranking |
| `sampling_function_source` | str | Same lambda as a string (used by AutoML code injection) |
| `extra_kwargs` | dict | Additional kwargs passed to prompt_class constructor |

**Note**: `gpu_id`, `seed`, `dataset_path`, `function_bank_path`, `k`, `k_word` are always passed automatically. Only put *additional* kwargs in `extra_kwargs`.

## Step 4: Expert Baselines

Create these files in `prompts/`:

### Postprocessing skeleton (required)

`prompts/your_task_expert_postprocessing_skeleton.py.txt` — A function template with the correct signature, docstring describing inputs/outputs, and placeholder logic. This is shown to the LLM as the postprocessing function API.

```python
def postprocess_preds(preds):
    """Describe the expected input format and output format.

    Args:
        preds: [describe prediction format from your tool.predict()]

    Returns:
        [describe expected output format]
    """
    import numpy as np
    # Placeholder logic
    return preds
```

### Expert postprocessing (optional)

`prompts/your_task_expert_postprocessing.py.txt` — A working expert implementation that can be used to seed the function bank as a starting point.

## Implicit Contracts to Know

### ImageData format

Images are **H, W, C** (height, width, channels). This is the convention for all tasks. If your tool expects a different format (e.g., C, H, W for PyTorch), convert inside your tool wrapper, not in the generated preprocessing functions.

### Generated function naming

The LLM must produce exactly `k` function pairs named:
- `preprocess_images_1`, `preprocess_images_2`, ..., `preprocess_images_k`
- `postprocess_preds_1`, `postprocess_preds_2`, ..., `postprocess_preds_k`

The execution template validates this at runtime. The agent system prompt enforces it via instructions. You don't need to handle this in your task code.

### Stateless execution

Each iteration runs as an independent Python script. Generated functions must:
- Import all dependencies inside the function body
- Not define inner/helper functions
- Not rely on any state from previous iterations

### Metric sorting

The `sampling_function` is used to rank functions in the bank. By default, **higher is better** (`maximize=True`). If your metric should be minimized, you'll need to negate it in the lambda or handle it in your prompt's function bank sampling.

## Testing Your Task

Before running a full experiment (which costs LLM API credits), verify your setup:

1. **Check template rendering**: Your prompt class should produce a valid Python script:
   ```python
   from prompts.your_task_prompts import YourTaskPrompts
   p = YourTaskPrompts(gpu_id=0, seed=42, dataset_path="/tmp/test",
                       function_bank_path="/tmp/bank.json", k=3, k_word="three")
   script = p.run_pipeline_prompt()
   # Verify no unfilled {placeholders} remain (except the codegen placeholder)
   import re
   unfilled = re.findall(r'\{[a-z_]+\}', script)
   print("Unfilled placeholders:", unfilled)  # Should only show the codegen placeholder
   ```

2. **Check your tool wrapper** independently:
   ```python
   from src.your_task import YourTool
   tool = YourTool(device=0)
   # Load a small test dataset and verify predict() + evaluate() work
   ```

3. **Run the full pipeline** with a small iteration count:
   ```bash
   python main.py -d /path/to/data --experiment_name your_task \
       --num_optim_iter 1 -k 1
   ```
