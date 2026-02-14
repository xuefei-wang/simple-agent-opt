"""Unit tests for main.function_bank_sample.

No LLM, GPU, or data files required.
"""
import json
import os
import tempfile
import unittest

from main import function_bank_sample


def _make_entry(f1_value):
    return {
        "preprocessing_function": f"def preprocess_images(images): return images  # f1={f1_value}",
        "postprocessing_function": "def postprocess_preds(preds): return preds",
        "overall_metrics": {"f1_score": f1_value},
    }


def _write_bank(entries):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(entries, f)
    return path


class TestFunctionBankSample(unittest.TestCase):

    def setUp(self):
        self.sorting_fn = lambda x: x["overall_metrics"]["f1_score"]
        self.entries = [_make_entry(v) for v in [0.1, 0.5, 0.9, 0.3, 0.7]]
        self.path = _write_bank(self.entries)

    def tearDown(self):
        os.unlink(self.path)

    def test_top_section_present(self):
        result = function_bank_sample(
            self.path, n_top=3, n_worst=0, n_last=0,
            sorting_function=self.sorting_fn,
            current_iteration=5, history_threshold=0,
        )
        self.assertIn("Top 3", result)

    def test_worst_section_present(self):
        result = function_bank_sample(
            self.path, n_top=0, n_worst=2, n_last=0,
            sorting_function=self.sorting_fn,
            current_iteration=5, history_threshold=0,
        )
        self.assertIn("Worst 2", result)

    def test_last_section_present(self):
        result = function_bank_sample(
            self.path, n_top=0, n_worst=0, n_last=2,
            sorting_function=self.sorting_fn,
            current_iteration=5, history_threshold=0,
        )
        self.assertIn("most recent 2", result)

    def test_all_sections_present(self):
        result = function_bank_sample(
            self.path, n_top=2, n_worst=2, n_last=2,
            sorting_function=self.sorting_fn,
            current_iteration=5, history_threshold=0,
        )
        self.assertIn("Top 2", result)
        self.assertIn("Worst 2", result)
        self.assertIn("most recent 2", result)

    def test_empty_bank(self):
        empty_path = _write_bank([])
        try:
            result = function_bank_sample(
                empty_path, n_top=3, n_worst=3, n_last=3,
                sorting_function=self.sorting_fn,
                current_iteration=5, history_threshold=0,
            )
            # Should not crash, should contain section headers
            self.assertIn("Top 3", result)
        finally:
            os.unlink(empty_path)

    def test_zero_counts_produce_empty_string(self):
        result = function_bank_sample(
            self.path, n_top=0, n_worst=0, n_last=0,
            sorting_function=self.sorting_fn,
            current_iteration=5, history_threshold=0,
        )
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
