import unittest
import numpy as np
import pytest
from src.spot_detection import DeepcellSpotsDetector
from src.data_io import ImageData

from dotenv import load_dotenv
'''
To run from project root directory: python -m tests.test_spotdetection
'''

@pytest.mark.integration
class TestSpotDetection(unittest.TestCase):
    def test_pipeline(self):
        '''Test if prediction and evaluation pipeline completes without errors'''
        load_dotenv()

        spots_data = np.load('../data/SpotNet-v1_1/val.npz', allow_pickle=True)

        images = ImageData(raw = spots_data['X'], batch_size = spots_data['X'].shape[0], image_ids = [i for i in range(spots_data['X'].shape[0])])
        spots_truth = spots_data['y']

        def preprocess_images(images: ImageData) -> ImageData:
            return images

        images = preprocess_images(images)

        deepcell_spot_detector = DeepcellSpotsDetector()

        # Predict spots
        pred = deepcell_spot_detector.predict(images)
        
        # Get metrics
        metrics = deepcell_spot_detector.evaluate(pred, spots_truth)
        
        print(metrics)
        
        self.assertIn('f1_score', metrics)


if __name__ == '__main__':
    unittest.main()