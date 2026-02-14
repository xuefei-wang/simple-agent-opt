"""
Module for loading and saving biological image data in various formats.
Provides framework-agnostic data structures and IO interfaces.
"""

import zarr
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ImageData:
    """Framework-agnostic container for batched image data. Handles variable
    image resolutions
    
    This class provides a standardized structure for storing and managing batched 
    image data along with related annotations and predictions.
    Data is internally converted to lists of arrays for flexibility with varying image sizes.

    The class accepts both lists of arrays and numpy arrays as input, but will convert them
    internally to lists to support variable-sized images across different frameworks.

    Attributes:
        raw (Union[List[np.ndarray], np.ndarray]): Raw image data, can be provided as either 
            a list of arrays or a numpy array. Each image should have shape (H, W, C).
        
        batch_size (Optional[int]): Number of images to include in the batch. Can be smaller 
            than the total dataset size. If None, will use the full dataset size.
        
        image_ids (Union[List[int], List[str], None]): Unique identifier(s) for images
            in the batch as a list. If None, auto-generated integer IDs [0,1,2,...] will be created.

        masks (Optional[Union[List[np.ndarray], np.ndarray]]): Ground truth segmentation masks.
            Integer-valued arrays where 0 is background and positive integers are unique 
            object identifiers. Each mask should have shape (H, W, 1) or (H, W).
        
        predicted_masks (Optional[Union[List[np.ndarray], np.ndarray]]): Model-predicted 
            segmentation masks. Each mask should have shape (H, W, 1) or (H, W).
        
        predicted_classes (Optional[List[Dict[int, str]]]): List of mappings from
            object identifiers to predicted classes for each image.
    """
    raw: Union[List[np.ndarray], np.ndarray]
    batch_size: Optional[int] = None
    image_ids: Optional[Union[int, str, List[Union[int, str]]]] = None
    masks: Optional[Union[List[np.ndarray], np.ndarray]] = None
    predicted_masks: Optional[Union[List[np.ndarray], np.ndarray]] = None
    predicted_classes: Optional[List[Dict[int, str]]] = None

    def __post_init__(self):
        """Validate and standardize data after initialization."""
        # Convert raw to a standardized format (list of arrays)
        if isinstance(self.raw, np.ndarray):
            if self.raw.dtype == object:
                # It's already an object array, convert to list
                self.raw = list(self.raw)
            else:
                # It's a regular ndarray, check if it's a batch
                if len(self.raw.shape) >= 3:  # Assuming (B,H,W,C) or similar
                    # Convert to list of arrays
                    self.raw = [self.raw[i] for i in range(self.raw.shape[0])]
                else:
                    # Single image
                    self.raw = [self.raw]
        # Validate raw data format
        for i, img in enumerate(self.raw):
            if len(img.shape) != 3:
                raise ValueError(f"Image at index {i} has {len(img.shape)} dimensions, expected 3 (H,W,C)")
            # Ensure all images have the same number of channels
            if i > 0 and img.shape[2] != self.raw[0].shape[2]:
                raise ValueError(f"Image at index {i} has {img.shape[2]} channels, expected {self.raw[0].shape[2]}")

        # Total dataset size
        total_size = len(self.raw)

        # Set batch size if not provided
        if self.batch_size is None:
            self.batch_size = total_size
        elif self.batch_size > total_size:
            self.batch_size = total_size
            # raise ValueError(f"Requested batch_size ({self.batch_size}) exceeds dataset size ({total_size})")        
        # Generate default image_ids if none provided - use integers [0,1,2,...]
        if self.image_ids is None:
            self.image_ids = list(range(total_size))
        elif not isinstance(self.image_ids, list):
            raise ValueError("image_ids must be a list of integers or strings")
        
        # Validate image_ids length matches total dataset size
        if len(self.image_ids) != total_size:
            raise ValueError(
                f"Number of image_ids ({len(self.image_ids)}) does not match dataset size ({total_size})"
            )
        
        # Apply similar conversion to masks
        if self.masks is not None:
            if isinstance(self.masks, np.ndarray):
                if self.masks.dtype == object:
                    self.masks = list(self.masks)
                else:
                    if len(self.masks.shape) >= 3:  # Batch
                        self.masks = [self.masks[i] for i in range(self.masks.shape[0])]
                    else:
                        # Single mask
                        self.masks = [self.masks]

            # Validate masks length
            if len(self.masks) != total_size:
                raise ValueError(f"Number of masks ({len(self.masks)}) does not match dataset size ({total_size})")
        
            # Check dimensions
            for i, (img, mask) in enumerate(zip(self.raw, self.masks)):
                # Ensure mask has either 2D or 3D shape
                if not (len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[2] == 1)):
                    raise ValueError(f"Mask at index {i} has invalid shape {mask.shape}, expected (H,W) or (H,W,1)")
                
                # Check spatial dimensions match (warn only — some tasks use mismatched sizes by design)
                if mask.shape[:2] != img.shape[:2]:
                    logging.warning(f"Mask shape {mask.shape[:2]} does not match image {i} dimensions {img.shape[:2]}")

        
        # Similar handling for predicted_masks
        if self.predicted_masks is not None:
            if isinstance(self.predicted_masks, np.ndarray):
                if self.predicted_masks.dtype == object:
                    self.predicted_masks = list(self.predicted_masks)
                else:
                    if len(self.predicted_masks.shape) >= 3:
                        self.predicted_masks = [self.predicted_masks[i] for i in range(self.predicted_masks.shape[0])]
                    else:
                        # Single mask
                        self.predicted_masks = [self.predicted_masks]
            
            # Validate predicted_masks length
            if len(self.predicted_masks) != total_size:
                raise ValueError(f"Number of predicted masks ({len(self.predicted_masks)}) does not match dataset size ({total_size})")

            # Validation
            for i, (img, pmask) in enumerate(zip(self.raw, self.predicted_masks)):
                # Ensure mask has either 2D or 3D shape
                if not (len(pmask.shape) == 2 or (len(pmask.shape) == 3 and pmask.shape[2] == 1)):
                    raise ValueError(f"Predicted mask at index {i} has invalid shape {pmask.shape}, expected (H,W) or (H,W,1)")
                
                # Check spatial dimensions match (warn only — some tasks use mismatched sizes by design)
                if pmask.shape[:2] != img.shape[:2]:
                    logging.warning(f"Predicted mask shape {pmask.shape[:2]} doesn't match image {i} dimensions {img.shape[:2]}")



        # Validate predicted_classes if provided
        if self.predicted_classes is not None and len(self.predicted_classes) != total_size:
            raise ValueError(
                f"Number of predicted_classes entries ({len(self.predicted_classes)}) "
                f"does not match dataset size ({total_size})"
            )

    @property
    def num_channels(self) -> int:
        """Get number of channels in raw image data."""
        # Get the number of channels from the first image
        return self.raw[0].shape[2]

    @property
    def spatial_shape(self) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
        """Get spatial dimensions (H, W) of images. Returns a tuple if all images have the same shape,
        otherwise returns a list of tuples.
        
        Returns one of:
            Tuple (H, W) if all images have the same shape,
            List of tuples (H, W) if images have different shapes
        """
        shapes = [img.shape[0:2] for img in self.raw]
        if all(shape == shapes[0] for shape in shapes):
            return shapes[0]
        else:
            return shapes

    
    @property
    def spatial_shapes(self) -> List[Tuple[int, int]]:
        """Get spatial dimensions (H, W) of images for each image in dataset.
        
        Returns:
            List of tuples (H, W) for each image in dataset
        """
        return [img.shape[0:2] for img in self.raw]
    
    def get_item(self, idx: int) -> 'ImageData':
        """Get single item from batch as new ImageData object.
        
        Args:
            idx: Index in batch to retrieve
            
        Returns:
            New ImageData object containing single item
        """
        if idx >= self.batch_size:
            raise IndexError(f"Index {idx} out of range for batch size {self.batch_size}")
            
        return ImageData(
            raw=[self.raw[idx]],
            batch_size=1,
            image_ids=[self.image_ids[idx]],
            masks=[self.masks[idx]] if self.masks is not None else None,
            predicted_masks=[self.predicted_masks[idx]] if self.predicted_masks is not None else None,
            predicted_classes=[self.predicted_classes[idx]] if self.predicted_classes is not None else None
        )
    
    def to_numpy(self) -> 'ImageDataNP':
        """Convert ImageData to ImageDataNP.  Beware, this won't handle datasets 
        with variable image sizes."""
        if isinstance(self.spatial_shape, list):
            raise ValueError("Cannot convert ImageData with variable image sizes to ImageDataNP")
        
        return ImageDataNP(
            raw=np.array(self.raw),
            masks=np.array(self.masks) if self.masks is not None else None,
            predicted_masks=np.array(self.predicted_masks) if self.predicted_masks is not None else None,
        )
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.raw)
    
    def __iter__(self) -> Iterator['ImageData']:
        """Iterate over items in dataset."""
        for i in range(len(self)):
            yield self.get_item(i)



@dataclass
class ImageDataNP:
    """stripped down ImageData class for numpy arrays"""
    raw: np.ndarray
    masks: Optional[np.ndarray] = None
    predicted_masks: Optional[np.ndarray] = None
    predicted_classes: Optional[np.ndarray] = None

    def __post_init__(self):
        """Convert lists to numpy arrays"""
        if isinstance(self.raw, list):
            self.raw = np.array(self.raw)
        if isinstance(self.masks, list):
            self.masks = np.array(self.masks)
        if isinstance(self.predicted_masks, list):
            self.predicted_masks = np.array(self.predicted_masks)

    def __len__(self) -> int:
        """Get dataset size."""
        return self.raw.shape[0]
    
    def __getitem__(self, idx: int) -> 'ImageDataNP':
        """Get item from dataset."""
        return ImageDataNP(
            raw=self.raw[idx],
            masks=self.masks[idx] if self.masks is not None else None,
            predicted_masks=self.predicted_masks[idx] if self.predicted_masks is not None else None,
            predicted_classes=self.predicted_classes[idx] if self.predicted_classes is not None else None
        )
