"""Unit tests for src/data_io.py — ImageData and ImageDataNP classes.

No LLM, GPU, or data files required.
"""
import unittest
import numpy as np

from src.data_io import ImageData, ImageDataNP


def _make_images(n, h=32, w=32, c=3, dtype=np.float32):
    """Helper: list of n random (H,W,C) arrays."""
    return [np.random.rand(h, w, c).astype(dtype) for _ in range(n)]


def _make_array(n, h=32, w=32, c=3, dtype=np.float32):
    """Helper: single (B,H,W,C) ndarray."""
    return np.random.rand(n, h, w, c).astype(dtype)


class TestImageDataConstruction(unittest.TestCase):
    """Construction from various input formats."""

    def test_from_list_of_arrays(self):
        imgs = _make_images(4)
        data = ImageData(raw=imgs)
        self.assertEqual(len(data), 4)
        self.assertEqual(data.batch_size, 4)

    def test_from_ndarray(self):
        arr = _make_array(4)
        data = ImageData(raw=arr)
        self.assertEqual(len(data), 4)
        self.assertIsInstance(data.raw, list)

    def test_from_object_array(self):
        imgs = _make_images(3, h=32, w=32)
        obj_arr = np.empty(3, dtype=object)
        for i, img in enumerate(imgs):
            obj_arr[i] = img
        data = ImageData(raw=obj_arr)
        self.assertEqual(len(data), 3)

    def test_single_image_in_list(self):
        # A single image must be wrapped in a list
        img = np.random.rand(32, 32, 3).astype(np.float32)
        data = ImageData(raw=[img])
        self.assertEqual(len(data), 1)

    def test_3d_ndarray_treated_as_batch(self):
        # A 3D ndarray (B,H,W) with shape[2]!=channel dim gets split along axis 0
        # A 4D ndarray (B,H,W,C) is the standard batch format
        arr = _make_array(4)  # (4, 32, 32, 3)
        data = ImageData(raw=arr)
        self.assertEqual(len(data), 4)


class TestBatchSizeClamping(unittest.TestCase):

    def test_batch_size_none_defaults_to_total(self):
        data = ImageData(raw=_make_images(5))
        self.assertEqual(data.batch_size, 5)

    def test_batch_size_larger_than_total_clamped(self):
        data = ImageData(raw=_make_images(3), batch_size=100)
        self.assertEqual(data.batch_size, 3)

    def test_batch_size_exact(self):
        data = ImageData(raw=_make_images(5), batch_size=5)
        self.assertEqual(data.batch_size, 5)

    def test_batch_size_smaller_than_total(self):
        data = ImageData(raw=_make_images(5), batch_size=2)
        self.assertEqual(data.batch_size, 2)


class TestImageIds(unittest.TestCase):

    def test_auto_generated_ids(self):
        data = ImageData(raw=_make_images(3))
        self.assertEqual(data.image_ids, [0, 1, 2])

    def test_custom_ids(self):
        data = ImageData(raw=_make_images(2), image_ids=["a", "b"])
        self.assertEqual(data.image_ids, ["a", "b"])

    def test_mismatched_ids_raises(self):
        with self.assertRaises(ValueError):
            ImageData(raw=_make_images(3), image_ids=["a"])

    def test_non_list_ids_raises(self):
        with self.assertRaises(ValueError):
            ImageData(raw=_make_images(1), image_ids="single_string")


class TestValidationErrors(unittest.TestCase):

    def test_wrong_dimensions_raises(self):
        bad_img = [np.random.rand(32, 32)]  # 2D, missing channel dim
        with self.assertRaises(ValueError):
            ImageData(raw=bad_img)

    def test_mismatched_channels_raises(self):
        img1 = np.random.rand(32, 32, 3).astype(np.float32)
        img2 = np.random.rand(32, 32, 1).astype(np.float32)
        with self.assertRaises(ValueError):
            ImageData(raw=[img1, img2])

    def test_mismatched_mask_count_raises(self):
        imgs = _make_images(3)
        masks = [np.zeros((32, 32, 1))]  # only 1 mask for 3 images
        with self.assertRaises(ValueError):
            ImageData(raw=imgs, masks=masks)

    def test_invalid_mask_shape_raises(self):
        imgs = _make_images(1)
        bad_mask = [np.zeros((32, 32, 3))]  # 3-channel mask is invalid
        with self.assertRaises(ValueError):
            ImageData(raw=imgs, masks=bad_mask)


class TestGetItem(unittest.TestCase):

    def test_get_item_returns_single_imagedata(self):
        data = ImageData(raw=_make_images(4), image_ids=["a", "b", "c", "d"])
        item = data.get_item(2)
        self.assertIsInstance(item, ImageData)
        self.assertEqual(len(item), 1)
        self.assertEqual(item.image_ids, ["c"])

    def test_get_item_with_masks(self):
        imgs = _make_images(2)
        masks = [np.zeros((32, 32, 1)) for _ in range(2)]
        data = ImageData(raw=imgs, masks=masks)
        item = data.get_item(1)
        self.assertIsNotNone(item.masks)
        self.assertEqual(len(item.masks), 1)

    def test_get_item_out_of_range_raises(self):
        data = ImageData(raw=_make_images(3), batch_size=2)
        with self.assertRaises(IndexError):
            data.get_item(2)


class TestLen(unittest.TestCase):

    def test_len_matches_num_images(self):
        data = ImageData(raw=_make_images(7))
        self.assertEqual(len(data), 7)


class TestIter(unittest.TestCase):

    def test_iter_yields_correct_count(self):
        data = ImageData(raw=_make_images(4))
        items = list(data)
        self.assertEqual(len(items), 4)
        for item in items:
            self.assertIsInstance(item, ImageData)
            self.assertEqual(len(item), 1)


class TestSpatialShape(unittest.TestCase):

    def test_uniform_shape_returns_tuple(self):
        data = ImageData(raw=_make_images(3, h=64, w=48))
        self.assertEqual(data.spatial_shape, (64, 48))

    def test_variable_shapes_returns_list(self):
        img1 = np.random.rand(32, 32, 3).astype(np.float32)
        img2 = np.random.rand(64, 48, 3).astype(np.float32)
        data = ImageData(raw=[img1, img2])
        shapes = data.spatial_shape
        self.assertIsInstance(shapes, list)
        self.assertEqual(shapes, [(32, 32), (64, 48)])


class TestSpatialShapes(unittest.TestCase):

    def test_returns_list_of_tuples(self):
        data = ImageData(raw=_make_images(3, h=16, w=24))
        shapes = data.spatial_shapes
        self.assertEqual(len(shapes), 3)
        self.assertTrue(all(s == (16, 24) for s in shapes))


class TestNumChannels(unittest.TestCase):

    def test_num_channels(self):
        data = ImageData(raw=_make_images(2, c=5))
        self.assertEqual(data.num_channels, 5)


class TestMaskConversion(unittest.TestCase):

    def test_ndarray_masks_converted_to_list(self):
        imgs = _make_images(3, h=16, w=16)
        masks_arr = np.zeros((3, 16, 16, 1))
        data = ImageData(raw=imgs, masks=masks_arr)
        self.assertIsInstance(data.masks, list)
        self.assertEqual(len(data.masks), 3)

    def test_predicted_masks_ndarray_converted(self):
        imgs = _make_images(2, h=16, w=16)
        pmasks = np.zeros((2, 16, 16, 1))
        data = ImageData(raw=imgs, predicted_masks=pmasks)
        self.assertIsInstance(data.predicted_masks, list)


class TestToNumpy(unittest.TestCase):

    def test_to_numpy_uniform(self):
        data = ImageData(raw=_make_images(3, h=16, w=16, c=3))
        np_data = data.to_numpy()
        self.assertIsInstance(np_data, ImageDataNP)
        self.assertEqual(np_data.raw.shape, (3, 16, 16, 3))

    def test_to_numpy_variable_raises(self):
        img1 = np.random.rand(32, 32, 3).astype(np.float32)
        img2 = np.random.rand(64, 48, 3).astype(np.float32)
        data = ImageData(raw=[img1, img2])
        with self.assertRaises(ValueError):
            data.to_numpy()


class TestImageDataNP(unittest.TestCase):

    def test_len(self):
        arr = _make_array(5)
        data = ImageDataNP(raw=arr)
        self.assertEqual(len(data), 5)

    def test_getitem(self):
        arr = _make_array(5, h=16, w=16, c=3)
        data = ImageDataNP(raw=arr)
        item = data[2]
        self.assertIsInstance(item, ImageDataNP)

    def test_list_input_converted(self):
        imgs = _make_images(3, h=16, w=16)
        data = ImageDataNP(raw=imgs)
        self.assertIsInstance(data.raw, np.ndarray)


if __name__ == "__main__":
    unittest.main()
