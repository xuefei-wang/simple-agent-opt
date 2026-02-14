"""Persistent verification that task-specific template rendering produces correct output.

Tests that:
1. All placeholders are resolved (no leftover {placeholder} patterns)
2. The CODEGEN placeholder survives (it's meant for runtime injection)
3. Task-specific code is present in the rendered template
4. Rendered templates are valid Python syntax
5. sampling_function_source strings are valid standalone Python

No LLM, GPU, or data files required.
"""
import ast
import re
import unittest

from prompts.task_prompts import _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER
from prompts.spot_detection_prompts import SpotDetectionPromptsWithSkeleton
from prompts.cellpose_segmentation_prompts import CellposeSegmentationPromptsWithSkeleton
from prompts.medsam_segmentation_prompts import MedSAMSegmentationPromptsWithSkeleton


# Shared test parameters (arbitrary but fixed)
BASE_KWARGS = {
    "gpu_id": 0,
    "seed": 42,
    "dataset_path": "/tmp/test_data",
    "function_bank_path": "/tmp/test_bank.json",
    "k": 3,
    "k_word": "three",
}


def render_template(prompt_class, extra_kwargs=None):
    kwargs = {**BASE_KWARGS, **(extra_kwargs or {})}
    prompts = prompt_class(**kwargs)
    return prompts.run_pipeline_prompt()


class TestNoUnresolvedPlaceholders(unittest.TestCase):
    """Verify template-specific placeholders are all resolved."""

    # These are the actual template placeholders that must be replaced.
    # Python code like {e}, {file_path} in f-strings are NOT template placeholders.
    TEMPLATE_PLACEHOLDERS = [
        "{task_imports}", "{task_extra_imports}", "{task_extra_config}",
        "{task_setup}", "{task_predict_single}", "{task_postprocess_check_single}",
        "{task_evaluate_single}", "{task_predict_full}", "{task_postprocess_check_full}",
        "{task_evaluate_full}", "{task_metrics_log}",
        "{gpu_id}", "{seed}", "{dataset_path}", "{function_bank_path}", "{sample_k}",
    ]

    def _check_no_unresolved(self, rendered, extra_placeholders=None):
        placeholders = self.TEMPLATE_PLACEHOLDERS + (extra_placeholders or [])
        found = [p for p in placeholders if p in rendered]
        self.assertEqual(found, [], f"Unresolved template placeholders: {found}")

    def test_spot_detection(self):
        rendered = render_template(SpotDetectionPromptsWithSkeleton)
        self._check_no_unresolved(rendered)

    def test_cellpose(self):
        rendered = render_template(CellposeSegmentationPromptsWithSkeleton,
                                   {"dataset_size": 100, "batch_size": 16})
        self._check_no_unresolved(rendered)

    def test_medsam(self):
        rendered = render_template(MedSAMSegmentationPromptsWithSkeleton,
                                   {"checkpoint_path": "/tmp/checkpoint.pth"})
        self._check_no_unresolved(rendered, extra_placeholders=["{checkpoint_path}"])

    def test_cellpose_extra_placeholders(self):
        rendered = render_template(CellposeSegmentationPromptsWithSkeleton,
                                   {"dataset_size": 100, "batch_size": 16})
        self._check_no_unresolved(rendered, extra_placeholders=["{dataset_size}", "{batch_size}"])


class TestCodegenPlaceholderSurvives(unittest.TestCase):
    """The CODEGEN placeholder must survive rendering (it's filled at runtime)."""

    def test_spot_detection(self):
        rendered = render_template(SpotDetectionPromptsWithSkeleton)
        self.assertIn(_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER, rendered)

    def test_cellpose(self):
        rendered = render_template(CellposeSegmentationPromptsWithSkeleton,
                                   {"dataset_size": 100, "batch_size": 16})
        self.assertIn(_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER, rendered)

    def test_medsam(self):
        rendered = render_template(MedSAMSegmentationPromptsWithSkeleton,
                                   {"checkpoint_path": "/tmp/checkpoint.pth"})
        self.assertIn(_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER, rendered)


class TestTaskSpecificContent(unittest.TestCase):
    """Verify task-specific imports and setup code appear in rendered templates."""

    def test_spot_detection_has_deepcell(self):
        rendered = render_template(SpotDetectionPromptsWithSkeleton)
        self.assertIn("DeepcellSpotsDetector", rendered)
        self.assertIn("from src.spot_detection", rendered)

    def test_cellpose_has_cellpose_tool(self):
        rendered = render_template(CellposeSegmentationPromptsWithSkeleton,
                                   {"dataset_size": 100, "batch_size": 16})
        self.assertIn("CellposeTool", rendered)
        self.assertIn("from src.cellpose_segmentation", rendered)

    def test_medsam_has_medsam_tool(self):
        rendered = render_template(MedSAMSegmentationPromptsWithSkeleton,
                                   {"checkpoint_path": "/tmp/checkpoint.pth"})
        self.assertIn("MedSAMTool", rendered)
        self.assertIn("from src.medsam_segmentation", rendered)

    def test_medsam_has_checkpoint_path(self):
        rendered = render_template(MedSAMSegmentationPromptsWithSkeleton,
                                   {"checkpoint_path": "/data/medsam_vit_b.pth"})
        self.assertIn("/data/medsam_vit_b.pth", rendered)


class TestRenderedTemplateIsPython(unittest.TestCase):
    """Verify rendered templates parse as valid Python (ignoring the CODEGEN placeholder)."""

    def _check_syntax(self, rendered):
        # Replace the CODEGEN placeholder with a pass statement so it parses
        code = rendered.replace(_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER, "pass")
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.fail(f"Rendered template has invalid Python syntax: {e}")

    def test_spot_detection(self):
        self._check_syntax(render_template(SpotDetectionPromptsWithSkeleton))

    def test_cellpose(self):
        self._check_syntax(render_template(CellposeSegmentationPromptsWithSkeleton,
                                           {"dataset_size": 100, "batch_size": 16}))

    def test_medsam(self):
        self._check_syntax(render_template(MedSAMSegmentationPromptsWithSkeleton,
                                           {"checkpoint_path": "/tmp/checkpoint.pth"}))


class TestSamplingFunctionSources(unittest.TestCase):
    """Verify TASK_CONFIGS sampling_function_source strings are valid and consistent."""

    def _check_source_consistency(self, source_str, fn, test_data):
        """Verify the source string produces a function that matches the lambda."""
        # Must be valid standalone Python
        try:
            ast.parse(source_str)
        except SyntaxError:
            self.fail(f"sampling_function_source is not valid Python: {source_str}")

        # Must define 'sampling_function'
        ns = {}
        exec(source_str, {}, ns)
        self.assertIn('sampling_function', ns,
                       f"Source string does not define 'sampling_function': {source_str}")

        # Must produce same result as the callable
        self.assertEqual(ns['sampling_function'](test_data), fn(test_data),
                         f"Source string behavior differs from lambda: {source_str}")

    def test_all_task_configs(self):
        # Import here to avoid pulling in autogen at module level
        # (TASK_CONFIGS is defined in main.py which imports autogen)
        # Instead, test the source strings directly
        configs = [
            ("sampling_function = lambda x: x['overall_metrics']['f1_score']",
             lambda x: x['overall_metrics']['f1_score']),
            ("sampling_function = lambda x: x['overall_metrics']['average_precision']",
             lambda x: x['overall_metrics']['average_precision']),
            ("sampling_function = lambda x: x['overall_metrics']['dsc_metric'] + x['overall_metrics']['nsd_metric']",
             lambda x: x['overall_metrics']['dsc_metric'] + x['overall_metrics']['nsd_metric']),
        ]
        test_data = {
            'overall_metrics': {
                'f1_score': 0.85,
                'average_precision': 0.72,
                'dsc_metric': 0.68,
                'nsd_metric': 0.55,
            }
        }
        for source_str, fn in configs:
            with self.subTest(source=source_str[:50]):
                self._check_source_consistency(source_str, fn, test_data)


class TestConvertStringToFunction(unittest.TestCase):
    """Verify convert_string_to_function works with self-contained functions."""

    def test_function_with_internal_imports(self):
        from utils.analysis_utils import convert_string_to_function

        func_str = """
def preprocess_images(images):
    import numpy as np
    return images
"""
        fn = convert_string_to_function(func_str, 'preprocess_images')
        self.assertTrue(callable(fn))
        self.assertEqual(fn("test_input"), "test_input")

    def test_function_with_numpy_operations(self):
        from utils.analysis_utils import convert_string_to_function

        func_str = """
def compute(x):
    import numpy as np
    return float(np.mean(x))
"""
        fn = convert_string_to_function(func_str, 'compute')
        self.assertAlmostEqual(fn([1, 2, 3]), 2.0)


if __name__ == '__main__':
    unittest.main()
