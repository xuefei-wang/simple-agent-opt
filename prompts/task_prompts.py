
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
import textwrap
import os

_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER = "# --- CODEGEN_PREPROCESSING_FUNCTIONS_INSERT ---"

@dataclass
class TaskPrompts:
    """Task prompts for a task."""

    gpu_id : int
    seed : int
    function_bank_path : str
    dataset_path : str

    dataset_info : str
    k : int
    k_word : str
    checkpoint_path : Optional[str] = None
    dataset_size : Optional[int] = None
    batch_size : Optional[int] = None

    def run_pipeline_prompt(self) -> str:
        """
        Reads the shared execution template, replaces task-specific and
        configuration placeholders, DEDENTS and STRIPS the result, and
        returns the script string containing the function placeholder.
        """
        template_file_path = os.path.join(os.path.dirname(__file__), "execution_template.py.txt")

        try:
            with open(template_file_path, 'r') as f:
                template_content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Execution template file not found at: {template_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading execution template file: {e}")

        # Task-specific replacements first (so nested placeholders like {gpu_id}
        # inside task_setup get resolved by the standard config pass afterward).
        replacement_values = self.get_template_replacements()
        replacement_values.update(self._get_standard_replacements())

        script = template_content
        for key, value in replacement_values.items():
            script = script.replace("{" + key + "}", value)

        return textwrap.dedent(script).strip()

    def _get_standard_replacements(self) -> dict:
        """Standard config replacements shared by all tasks."""
        replacements = {
            "gpu_id": str(self.gpu_id),
            "seed": str(self.seed),
            "dataset_path": self.dataset_path.replace("\\", "/"),
            "function_bank_path": self.function_bank_path.replace("\\", "/"),
            "_PREPROCESSING_POSTPROCESSING_FUNCTIONS_PLACEHOLDER": _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER,
            "sample_k": str(self.k),
        }
        if self.checkpoint_path is not None:
            replacements["checkpoint_path"] = self.checkpoint_path.replace("\\", "/")
        if self.dataset_size is not None:
            replacements["dataset_size"] = str(self.dataset_size)
        if self.batch_size is not None:
            replacements["batch_size"] = str(self.batch_size)
        return replacements

    @abstractmethod
    def get_template_replacements(self) -> dict:
        """Return task-specific template replacement values."""
        pass

    @abstractmethod
    def get_task_details(self):
        pass

    @abstractmethod
    def get_pipeline_metrics_info(self):
        pass

    def get_postprocessing_function_api(self):
        filename = getattr(self, 'postprocessing_skeleton_filename', None)
        if not filename:
            return None
        api_file_path = os.path.join(os.path.dirname(__file__), filename)
        with open(api_file_path, 'r') as f:
            return textwrap.dedent(f.read())
