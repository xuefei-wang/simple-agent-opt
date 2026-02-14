import unittest
import argparse
import sys
import os
import pickle
import pytest

from src.medsam_segmentation import MedSAMTool
from src.data_io import ImageData

'''
To run from the project root directory:

python -m tests.test_medsam_segmentation --data_path {data_path} --checkpoint_path {pth_path} --device {gpu_id}
'''

@pytest.mark.integration
class TestMedSAMSegmentation(unittest.TestCase):
    def test_pipeline(self):
        """
        Tests that the prediction and evaluation pipeline completes without errors.
        """

        unpacked_info_path = os.path.join(self.data_path, "unpacked_info.pkl")
        resized_imgs_path = os.path.join(self.data_path, "resized_imgs.pkl")

        with open(unpacked_info_path, "rb") as f:
            _, _, used_val_raw_gts = pickle.load(f)

        with open(resized_imgs_path, "rb") as f:
            imgs, boxes = pickle.load(f)

        images = ImageData(raw=imgs,
            batch_size=8,
            image_ids=[i for i in range(len(imgs))],
            masks=used_val_raw_gts,
            predicted_masks=used_val_raw_gts)

        segmenter = MedSAMTool(gpu_id=self.device, checkpoint_path=self.checkpoint_path)
        pred_masks = segmenter.predict(images, boxes, used_for_baseline=False)
        metrics = segmenter.evaluate(pred_masks, images.masks)

        self.assertIn('dsc_metric', metrics)
        self.assertIn('nsd_metric', metrics)

def parse_args():
    parser = argparse.ArgumentParser(description='Run MedSAM segmentation tests with custom parameters')
    parser.add_argument('--device', type=int, default=0,
                      help='GPU device ID to run on.')
    parser.add_argument('--checkpoint_path', type=str, default='../data/medsam_vit_b.pth', help='Path to the model checkpoint file.')
    parser.add_argument('--data_path', type=str, default='../data/medsam_exp_data', help='Path to dataset.')
    
    # This allows unittest arguments to be passed through
    args, unittest_args = parser.parse_known_args()
    sys.argv[1:] = unittest_args
    
    return args

if __name__ == '__main__':
    args = parse_args()
    TestMedSAMSegmentation.device = args.device
    TestMedSAMSegmentation.checkpoint_path = args.checkpoint_path
    TestMedSAMSegmentation.data_path = args.data_path
    unittest.main()
