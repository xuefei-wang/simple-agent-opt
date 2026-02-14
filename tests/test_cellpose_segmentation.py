import unittest
import numpy as np
import argparse
import sys
import pytest
from src.cellpose_segmentation import CellposeTool
from src.data_io import ImageData
from cellpose.io import imread
from dotenv import load_dotenv
import glob
from numpy.testing import assert_array_equal
'''
To run from project root directory: python -m tests.test_cellpose_segmentation
'''

@pytest.mark.integration
class TestCellposeSegmentation(unittest.TestCase):
    # Class variables to store command line arguments
    data_path = '../data/cellpose/my_split/val_set/'
    device = 1
    model_name = "cyto3"
    num_files = 8

    def test_pipeline(self):
        '''Test if prediction and evaluation pipeline completes without errors'''
        load_dotenv()

        segmenter = CellposeTool(model_name='cyto3', device=self.device)
        raw_images, gt_masks = segmenter.loadData(self.data_path)
        raw_images = raw_images[:self.num_files]
        gt_masks = gt_masks[:self.num_files]

        images = ImageData(raw=raw_images,
                        batch_size=8,
                        image_ids=[i for i in range(self.num_files)]) 
        
        def preprocess_images(images: ImageData) -> ImageData:
            return images

        images = preprocess_images(images)        

        pred_masks = segmenter.predict(images, batch_size=images.batch_size)
        metrics, losses = segmenter.evaluate(pred_masks, gt_masks)
        print(metrics)
        print(losses)
            
        self.assertIn('average_precision', metrics)
        self.assertIn('bce_loss', losses)
   


def parse_args():
    parser = argparse.ArgumentParser(description='Run Cellpose segmentation tests with custom parameters')
    parser.add_argument('--data_path', type=str, default='../data/cellpose/my_split/val_set/',
                      help='Path to the dataset directory')
    parser.add_argument('--device', type=int, default=1,
                      help='GPU ID to use.')
    parser.add_argument('--num_files', type=int, default=8,
                      help='Number of files to test')
    
    # This allows unittest arguments to be passed through
    args, unittest_args = parser.parse_known_args()
    sys.argv[1:] = unittest_args
    
    return args


if __name__ == '__main__':
    # Parse arguments and set class variables
    args = parse_args()
    TestCellposeSegmentation.data_path = args.data_path
    TestCellposeSegmentation.device = args.device
    TestCellposeSegmentation.num_files = args.num_files
    
    unittest.main()