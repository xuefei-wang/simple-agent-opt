"""Unit tests for utils/function_bank_utils.py.

No LLM, GPU, or data files required. Uses temp JSON files.
"""
import json
import os
import tempfile
import unittest

from utils.function_bank_utils import (
    should_include_function,
    top_n,
    worst_n,
    last_n,
    pretty_print_list,
)


def _make_entry(metric_value, automl_optimized=None, automl_superseded=None):
    """Helper: create a function bank entry with given metric and automl flags."""
    entry = {
        "preprocessing_function": "def preprocess_images(images): return images",
        "postprocessing_function": "def postprocess_preds(preds): return preds",
        "overall_metrics": {"f1_score": metric_value},
    }
    if automl_optimized is not None:
        entry["automl_optimized"] = automl_optimized
    if automl_superseded is not None:
        entry["automl_superseded"] = automl_superseded
    return entry


def _write_bank(entries):
    """Write entries to a temp JSON file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(entries, f)
    return path


class TestShouldIncludeFunction(unittest.TestCase):

    def test_optimized_included(self):
        entry = _make_entry(0.9, automl_optimized=True)
        self.assertTrue(should_include_function(entry))

    def test_superseded_excluded(self):
        entry = _make_entry(0.9, automl_superseded=True)
        self.assertFalse(should_include_function(entry))

    def test_failed_optimization_excluded(self):
        entry = _make_entry(0.5, automl_optimized=False)
        self.assertFalse(should_include_function(entry))

    def test_not_improved_included(self):
        # automl_superseded=False means "we tried but original was kept"
        entry = _make_entry(0.7, automl_superseded=False)
        self.assertTrue(should_include_function(entry))

    def test_never_optimized_included(self):
        entry = _make_entry(0.6)
        self.assertTrue(should_include_function(entry))


class TestTopN(unittest.TestCase):

    def setUp(self):
        self.sorting_fn = lambda x: x["overall_metrics"]["f1_score"]

    def test_returns_top_entries(self):
        entries = [_make_entry(v) for v in [0.3, 0.9, 0.1, 0.7, 0.5]]
        path = _write_bank(entries)
        try:
            result = top_n(path, self.sorting_fn, n=3)
            values = [self.sorting_fn(e) for e in result]
            self.assertEqual(values, [0.9, 0.7, 0.5])
        finally:
            os.unlink(path)

    def test_excludes_superseded(self):
        entries = [
            _make_entry(0.9, automl_superseded=True),
            _make_entry(0.5),
            _make_entry(0.3),
        ]
        path = _write_bank(entries)
        try:
            result = top_n(path, self.sorting_fn, n=3)
            self.assertEqual(len(result), 2)
            self.assertEqual(self.sorting_fn(result[0]), 0.5)
        finally:
            os.unlink(path)

    def test_empty_bank(self):
        path = _write_bank([])
        try:
            result = top_n(path, self.sorting_fn, n=5)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)

    def test_n_larger_than_bank(self):
        entries = [_make_entry(0.5), _make_entry(0.3)]
        path = _write_bank(entries)
        try:
            result = top_n(path, self.sorting_fn, n=10)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(path)

    def test_none_metrics_filtered(self):
        entries = [_make_entry(0.5)]
        entries.append({
            "preprocessing_function": "...",
            "postprocessing_function": "...",
            "overall_metrics": {"f1_score": None},
        })
        path = _write_bank(entries)
        try:
            result = top_n(path, self.sorting_fn, n=5)
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(path)


class TestWorstN(unittest.TestCase):

    def setUp(self):
        self.sorting_fn = lambda x: x["overall_metrics"]["f1_score"]

    def test_returns_worst_entries(self):
        entries = [_make_entry(v) for v in [0.3, 0.9, 0.1, 0.7, 0.5]]
        path = _write_bank(entries)
        try:
            result = worst_n(path, self.sorting_fn, n=2)
            values = [self.sorting_fn(e) for e in result]
            self.assertEqual(values, [0.1, 0.3])
        finally:
            os.unlink(path)

    def test_empty_bank(self):
        path = _write_bank([])
        try:
            result = worst_n(path, self.sorting_fn, n=5)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)


class TestLastN(unittest.TestCase):

    def test_returns_last_entries(self):
        entries = [_make_entry(v) for v in [0.1, 0.2, 0.3, 0.4, 0.5]]
        path = _write_bank(entries)
        try:
            result = last_n(path, n=2)
            values = [e["overall_metrics"]["f1_score"] for e in result]
            self.assertEqual(values, [0.4, 0.5])
        finally:
            os.unlink(path)

    def test_excludes_superseded(self):
        entries = [
            _make_entry(0.1),
            _make_entry(0.9, automl_superseded=True),
            _make_entry(0.3),
        ]
        path = _write_bank(entries)
        try:
            result = last_n(path, n=5)
            self.assertEqual(len(result), 2)
        finally:
            os.unlink(path)

    def test_empty_bank(self):
        path = _write_bank([])
        try:
            result = last_n(path, n=5)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)


class TestPrettyPrintList(unittest.TestCase):

    def test_empty_list(self):
        result = pretty_print_list([])
        self.assertIn("No entries", result)

    def test_formats_entries(self):
        entries = [_make_entry(0.85)]
        result = pretty_print_list(entries)
        self.assertIn("Entry 1", result)
        self.assertIn("f1_score", result)
        self.assertIn("0.85", result)
        self.assertIn("preprocess_images", result)

    def test_multiple_entries(self):
        entries = [_make_entry(0.5), _make_entry(0.9)]
        result = pretty_print_list(entries)
        self.assertIn("Entry 1", result)
        self.assertIn("Entry 2", result)


if __name__ == "__main__":
    unittest.main()
