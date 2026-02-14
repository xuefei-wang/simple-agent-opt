"""Unit tests for main.save_seed_list.

No LLM, GPU, or data files required.
"""
import os
import tempfile
import unittest

from main import save_seed_list


class TestSaveSeedList(unittest.TestCase):

    def test_deterministic_same_seed(self):
        """Same initial seed always produces identical seed list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "seeds1.txt")
            path2 = os.path.join(tmpdir, "seeds2.txt")
            seeds1 = save_seed_list(10, path1, initial_seed=42)
            seeds2 = save_seed_list(10, path2, initial_seed=42)
            self.assertEqual(seeds1, seeds2)

    def test_different_seeds_differ(self):
        """Different initial seeds produce different lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "seeds1.txt")
            path2 = os.path.join(tmpdir, "seeds2.txt")
            seeds1 = save_seed_list(10, path1, initial_seed=42)
            seeds2 = save_seed_list(10, path2, initial_seed=99)
            self.assertNotEqual(seeds1, seeds2)

    def test_correct_count(self):
        """Generated list has the requested number of seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "seeds.txt")
            seeds = save_seed_list(20, path, initial_seed=1)
            self.assertEqual(len(seeds), 20)

    def test_seeds_are_integers(self):
        """All generated seeds are integers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "seeds.txt")
            seeds = save_seed_list(5, path, initial_seed=7)
            for s in seeds:
                self.assertIsInstance(s, int)

    def test_file_written(self):
        """Seeds are written to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "seeds.txt")
            seeds = save_seed_list(5, path, initial_seed=42)
            with open(path) as f:
                lines = f.read().strip().split("\n")
            self.assertEqual(len(lines), 5)
            file_seeds = [int(line) for line in lines]
            self.assertEqual(file_seeds, seeds)


if __name__ == "__main__":
    unittest.main()
