"""Unit tests for utils/executors.py — TemplatedLocalCommandLineCodeExecutor.

Tests the template injection logic without actually executing code.
No LLM, GPU, or data files required.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from autogen.coding import CodeBlock

from utils.executors import TemplatedLocalCommandLineCodeExecutor


PLACEHOLDER = "# --- CODEGEN_PREPROCESSING_FUNCTIONS_INSERT ---"

TEMPLATE_WITH_PLACEHOLDER = f"""\
import numpy as np
{PLACEHOLDER}
print("done")
"""

TEMPLATE_WITHOUT_PLACEHOLDER = """\
import numpy as np
print("done")
"""


def _make_executor(template_text=TEMPLATE_WITH_PLACEHOLDER, placeholder=PLACEHOLDER):
    """Create an executor with a mock template function."""
    return TemplatedLocalCommandLineCodeExecutor(
        template_script_func=lambda: template_text,
        placeholder=placeholder,
        work_dir="/tmp/claude",
    )


class TestPlaceholderReplacement(unittest.TestCase):

    @patch.object(TemplatedLocalCommandLineCodeExecutor, '__init__', lambda self, **kw: None)
    def test_placeholder_replaced(self):
        """Placeholder is replaced with generated code in the final script."""
        executor = TemplatedLocalCommandLineCodeExecutor.__new__(TemplatedLocalCommandLineCodeExecutor)
        executor._template_script_func = lambda: TEMPLATE_WITH_PLACEHOLDER
        executor._placeholder = PLACEHOLDER

        code = "def preprocess_images_1(images): return images"
        code_blocks = [CodeBlock(language="python", code=code)]

        # Mock the parent's execute_code_blocks to capture the injected script
        with patch("autogen.coding.LocalCommandLineCodeExecutor.execute_code_blocks") as mock_exec:
            mock_exec.return_value = (0, "success", None)
            executor.execute_code_blocks(code_blocks)
            # Check the code block passed to parent
            call_args = mock_exec.call_args[0][0]
            injected_code = call_args[0].code
            self.assertIn("def preprocess_images_1(images): return images", injected_code)
            self.assertNotIn(PLACEHOLDER, injected_code)


class TestMissingPlaceholder(unittest.TestCase):

    def test_missing_placeholder_returns_error(self):
        """Missing placeholder returns error SimpleNamespace."""
        executor = _make_executor(template_text=TEMPLATE_WITHOUT_PLACEHOLDER)
        code_blocks = [CodeBlock(language="python", code="x = 1")]
        result = executor.execute_code_blocks(code_blocks)
        self.assertIsInstance(result, SimpleNamespace)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Placeholder", result.output)


class TestMultipleCodeBlocks(unittest.TestCase):

    def test_multiple_blocks_rejected(self):
        """More than one code block returns error."""
        executor = _make_executor()
        code_blocks = [
            CodeBlock(language="python", code="x = 1"),
            CodeBlock(language="python", code="y = 2"),
        ]
        result = executor.execute_code_blocks(code_blocks)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Expected exactly 1", result.output)


class TestNonPythonBlocks(unittest.TestCase):

    def test_non_python_rejected(self):
        """Non-Python code block returns error."""
        executor = _make_executor()
        code_blocks = [CodeBlock(language="bash", code="echo hello")]
        result = executor.execute_code_blocks(code_blocks)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("python", result.output.lower())


class TestTemplateRetrievalError(unittest.TestCase):

    def test_template_func_error_handled(self):
        """Exception in template function is caught and returned as error."""
        def broken_template():
            raise RuntimeError("template broken")

        executor = TemplatedLocalCommandLineCodeExecutor(
            template_script_func=broken_template,
            placeholder=PLACEHOLDER,
            work_dir="/tmp/claude",
        )
        code_blocks = [CodeBlock(language="python", code="x = 1")]
        result = executor.execute_code_blocks(code_blocks)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("template", result.output.lower())


if __name__ == "__main__":
    unittest.main()
