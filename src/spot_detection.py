from typing import Optional
import os
from deepcell_spots.applications import SpotDetection
from deepcell_spots.dotnet_losses import DotNetLosses
from deepcell_spots.utils.augmentation_utils import subpixel_distance_transform
from deepcell_spots.utils.postprocessing_utils import y_annotations_to_point_list_max
from deepcell_spots.point_metrics import point_F1_score
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf


from src.data_io import ImageData


class DeepcellSpotsDetector():
    """Abstract base class defining the interface for cell spot detection models.

    This class provides a standardized interface that all spot detection model
    implementations must follow. It defines the core methods needed for image
    preprocessing and cell segmentation prediction while remaining model-agnostic.

    All implementations must handle batched inputs and outputs consistently,
    following the shape conventions defined in the ImageData class.

    Methods:
        preprocess: Prepare raw image data for model input
        predict: Generate cell spot predictions for the input images
        evaluate: 

    Example Implementation:
        >>> class MySpotDetector(BaseSpotDetector):
        ...     def preprocess(self, image_data):
        ...         # Implementation
        ...         pass
        ...
        ...     def predict(self, image_data):
        ...         # Implementation
        ...         pass

    Notes:
        - All methods must preserve batch dimensions
        - Implementations should handle both single-channel and multi-channel inputs
        - Error handling should be comprehensive and informative
        - Memory efficiency should be considered for large batches
    """
    
    def __init__(self, model_name='deepcell_spots', **kwargs):
        self.kwargs = kwargs
    
    def predict(self, images: ImageData) -> np.ndarray:
        """
        Detect spots in the given image using the loaded model.

        Parameters:
        - image: numpy array of shape (H, W, C) representing the input image.

        Returns:
        Predictions with two keys ('classification', 'offset_regression')
        - classification prediction: image array of one-hot encoded classifications
        - regression prediction: image array of regression distance from predicted points
    """
        home_dir = os.path.expanduser("~")
        model_path = os.path.join(home_dir, '.deepcell/models/SpotDetection-8')
        model = tf.keras.models.load_model(
            model_path, custom_objects={
                'regression_loss': DotNetLosses.regression_loss,
                'classification_loss': DotNetLosses.classification_loss
            }
        )
        app = SpotDetection(model)
        
        # Disable default preprocessing and postprocessing
        app.preprocessing_fn = None
        app.postprocessing_fn = None

        pred = app.predict(images.to_numpy().raw, batch_size=images.batch_size, threshold=0.95)

        return pred
    
    def evaluate(self, pred: np.ndarray, truth: np.ndarray) -> dict:
        """
        Evaluate detections against ground truth using the model output.

        Parameters:
            - pred: ndarray, list of  x and y coordinates of predicted spots per image
            - truth: ndarray, list of x and y coordinates of true spots per image

        Returns:
            Dict[str, float]: Dictionary with metric(s):
                - f1_score: Mean F1 score of predicted spots
                - class_loss: Mean IoU of correctly matched objects
                - regress_loss: Fraction of predicted objects that match ground truth
        """
        f1_list = []
        for p, t in zip(pred, truth):
            f1_list.append(point_F1_score(p, t, threshold=1))

        return {
            "f1_score": np.mean(f1_list),
        }