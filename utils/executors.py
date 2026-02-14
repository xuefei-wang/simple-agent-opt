import logging
from typing import List, Tuple, Optional, Callable, Any
from types import SimpleNamespace
import traceback
import json

from autogen.coding import LocalCommandLineCodeExecutor, CodeBlock

class TemplatedLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    """
    An executor that injects generated Python code into a template script
    before execution using LocalCommandLineCodeExecutor. Returns a SimpleNamespace
    mimicking the expected result structure for the agent.
    """
    def __init__(
        self,
        template_script_func: Callable[[], str],
        placeholder: str,
        **kwargs: Any,
    ):
        """
        Initializes the templated executor.

        Args:
            template_script_func: A callable function that returns the template
                                  script string (e.g., prompts.run_pipeline_prompt).
            placeholder: The exact string within the template to replace with
                         the generated code.
            **kwargs: Additional arguments passed to the parent
                      LocalCommandLineCodeExecutor (e.g., timeout, work_dir).
        """
        super().__init__(**kwargs)
        if not callable(template_script_func):
            raise ValueError("template_script_func must be a callable function.")
        self._template_script_func = template_script_func
        self._placeholder = placeholder


    def execute_code_blocks(
        self,
        code_blocks: List[CodeBlock],
    ) -> SimpleNamespace:
        """
        Overrides the default execution. Expects one Python code block,
        injects its content AND the code string itself into the template,
        executes the result, and returns a SimpleNamespace object.
        """
        # --- Input Validation ---
        if len(code_blocks) != 1:
            error_msg = f"Error: Expected exactly 1 code block, received {len(code_blocks)}."

            return SimpleNamespace(exit_code=1, output=error_msg)

        first_block = code_blocks[0]
        if first_block.language not in ["python", "py"]:
            error_msg = (
                f"Error: Expected first code block language to be 'python' or 'py', "
                f"got '{first_block.language}'."
            )

            return SimpleNamespace(exit_code=1, output=error_msg)

        generated_code = first_block.code

        # --- Template Retrieval ---
        try:
            template_script = self._template_script_func()
        except Exception as e:
            error_msg = f"Error: Failed to retrieve template script: {e}"
            # Return SimpleNamespace indicating failure
            return SimpleNamespace(exit_code=1, output=f"{error_msg}\n{traceback.format_exc()}")

        # --- Placeholder Check ---
        if self._placeholder not in template_script:
            error_msg = (
                f"Error: Placeholder '{self._placeholder}' not found in the template script."
            )
            # Return SimpleNamespace indicating failure
            return SimpleNamespace(exit_code=1, output=error_msg)

        # --- Code Injection (Function Definition) ---
        try:
            # Replace the main placeholder with the function code
            script_with_func = template_script.replace(self._placeholder, generated_code)
        except Exception as e:
            error_msg = f"Error: Failed during function placeholder replacement: {e}"
            # logger.exception(error_msg)
            # Return SimpleNamespace indicating failure
            return SimpleNamespace(exit_code=1, output=f"{error_msg}\n{traceback.format_exc()}")

        # --- Execution via Parent Class ---
        final_code_block = CodeBlock(language="python", code=script_with_func)

        # Call the original execute_code_blocks with the single, combined script
        exit_code, output, image = super().execute_code_blocks([final_code_block])

        result_obj = SimpleNamespace(exit_code=exit_code, output=output)

        return result_obj # Return the SimpleNamespace object
