from src.cellpose_segmentation import CellposeTool
from src.data_io import ImageData
import os
import numpy as np
import matplotlib.pyplot as plt

from cellpose.io import imread
import glob
import argparse




def minmaxnorm_tissuenet(img_array: np.ndarray) -> np.ndarray:
    """ Vectorized min-max normalization for tissuenet images.
        Normalizes each channel of each image independently.
        Then adds zeros as the third channel.
    """
    # Get batch size
    batch_size = img_array.shape[0]
    
    # Create output array with same shape as input
    normed_array = np.zeros_like(img_array, dtype=np.float64)
    
    # Loop through each image in the batch
    for i in range(batch_size):
        # For each channel (0 and 1)
        for c in range(img_array.shape[3]):
            # Extract this image's channel
            channel = img_array[i, :, :, c]
            
            # Calculate min and max for this specific image and channel
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            # Prevent division by zero (if the channel is constant)
            if max_val > min_val:
                normed_array[i, :, :, c] = (channel - min_val) / (max_val - min_val)
            else:
                # If max equals min, set all values to 0 or 0.5 to avoid NaN
                normed_array[i, :, :, c] = 0.0
    
    # Add zeros as the third channel
    normed_array = np.concatenate([normed_array, np.zeros_like(normed_array[:, :, :, :1])], axis=-1)
    
    return normed_array

def load_tissuenet_data(data_dir: str):
    """Load tissuenet data and normalize it, add np.zeros to last channel. Select cytoplasmic label"""
    test_dict = np.load(os.path.join(data_dir, 'tissuenet_v1.1_test.npz'))
    test_X, test_y = test_dict['X'], test_dict['y'] # test_y: cell = 0, nucleus = 1. we want to use cell
    
    normed_tissuenet_images = minmaxnorm_tissuenet(test_X)
    test_y = test_y[:,:,:,0]
    # keep the phantom dimensions
    test_y = test_y[:,:,:,np.newaxis]
    # Convert to a list of length b
    normed_tissuenet_images = [normed_tissuenet_images[i] for i in range(normed_tissuenet_images.shape[0])]
    test_y = [test_y[i] for i in range(test_y.shape[0])]
    
    return normed_tissuenet_images, test_y



def load_and_normalize_bact_fluor(dir_path: str) -> np.ndarray:
    """Normalize the image to 0-1 and return a list of images and masks, vectorized"""
    files = glob.glob(os.path.join(dir_path, '**', '*.tif'), recursive=True)
    files = sorted([file for file in files if 'masks' not in file])
    imgs = [imread(file).astype(np.float32) for file in files]
    masks = [imread(file.split('.')[0] + '_masks.tif') for file in files]
    masks = [mask[:,:,np.newaxis] for mask in masks]
    #  do per image min-max normalization of imgs a la (img_array - img_array.min(axis=(1,2,3), keepdims=True)) / (img_array.max(axis=(1,2,3), keepdims=True) - img_array.min(axis=(1,2,3), keepdims=True))
    new_imgs = []
    for img in imgs:
        img = (img - img.min()) / (img.max() - img.min()) # img shape: (h, w)
        # add np.zeros to first channel and last channel and add phantom dim for img
        img = img[:,:,np.newaxis]

        img = np.concatenate([np.zeros_like(img[:,:,:1]), img, np.zeros_like(img[:,:,:1])], axis=-1)
        new_imgs.append(img)

    return new_imgs, masks



# glob load recursively all .tif files in directory and all subdirectories
def load_and_normalize_bact_phase(dir_path: str) -> np.ndarray:
    """Normalize the image to 0-1 and return a list of images and masks, vectorized"""
    files = glob.glob(os.path.join(dir_path, '**', '*.tif'), recursive=True)
    files = [f for f in files if 'flows' not in f]
    files = sorted([file for file in files if 'masks' not in file])
    imgs = [imread(file).astype(np.float32) for file in files]
    masks = [imread(file.split('.')[0] + '_masks.tif') for file in files]
    masks = [mask[:,:,np.newaxis] for mask in masks]
    #  do per image min-max normalization of imgs a la (img_array - img_array.min(axis=(1,2,3), keepdims=True)) / (img_array.max(axis=(1,2,3), keepdims=True) - img_array.min(axis=(1,2,3), keepdims=True))
    new_imgs = []
    for img in imgs:
        img = (img - img.min()) / (img.max() - img.min()) # img shape: (h, w)
        # add np.zeros to first channel and last channel and add phantom dim for img
        img = img[:,:,np.newaxis]

        img = np.concatenate([np.zeros_like(img[:,:,:1]), img, np.zeros_like(img[:,:,:1])], axis=-1)
        new_imgs.append(img)

    return new_imgs, masks


def main(data_dir: str, save_dir: str, to_save: bool):
    segmenter = CellposeTool()
    # Instead, let's do a balanced split.  
    # Shuffle each dataset separately, 
    # equally split each into val/test.  Then combine each val and test set
    # Cellpose
    cp_images, cp_masks = segmenter.loadData(os.path.join(data_dir, 'cellpose/test/'))
    # Bact phase
    bp_images, bp_masks = load_and_normalize_bact_phase(os.path.join(data_dir, 'bact_phase', 'test_sorted'))
    # Bact fluor
    bf_images, bf_masks = load_and_normalize_bact_fluor(os.path.join(data_dir, 'bact_fluor', 'test_sorted'))
    # Tissuenet
    tn_images, tn_masks = load_tissuenet_data(data_dir)
    np.random.seed(42)

    # shuffle each dataset separately. Ensure that the masks are shuffled in the same way as the images
    #cp
    cp_perm_idx = np.random.permutation(np.arange(len(cp_images)))
    cp_images = [cp_images[i] for i in cp_perm_idx]
    cp_masks = [cp_masks[i] for i in cp_perm_idx]

    # #bp
    bp_perm_idx = np.random.permutation(np.arange(len(bp_images)))
    bp_images = [bp_images[i] for i in bp_perm_idx]
    bp_masks = [bp_masks[i] for i in bp_perm_idx]

    # bf
    bf_perm_idx = np.random.permutation(np.arange(len(bf_images)))
    bf_images = [bf_images[i] for i in bf_perm_idx]
    bf_masks = [bf_masks[i] for i in bf_perm_idx]

    # tn
    tn_perm_idx = np.random.permutation(np.arange(len(tn_images)))
    tn_images = [tn_images[i] for i in tn_perm_idx]
    tn_masks = [tn_masks[i] for i in tn_perm_idx]

    # Now let's split each dataset into val and test 50 50
    #cp
    cp_val_images = cp_images[:len(cp_images)//2]
    cp_val_masks = cp_masks[:len(cp_masks)//2]
    cp_test_images = cp_images[len(cp_images)//2:]
    cp_test_masks = cp_masks[len(cp_masks)//2:]

    #bp
    bp_val_images = bp_images[:len(bp_images)//2]
    bp_val_masks = bp_masks[:len(bp_masks)//2]
    bp_test_images = bp_images[len(bp_images)//2:]
    bp_test_masks = bp_masks[len(bp_masks)//2:]

    #bf
    bf_val_images = bf_images[:len(bf_images)//2]
    bf_val_masks = bf_masks[:len(bf_masks)//2]
    bf_test_images = bf_images[len(bf_images)//2:]
    bf_test_masks = bf_masks[len(bf_masks)//2:]

    #tn
    tn_val_images = tn_images[:len(tn_images)//2]
    tn_val_masks = tn_masks[:len(tn_masks)//2]
    tn_test_images = tn_images[len(tn_images)//2:]
    tn_test_masks = tn_masks[len(tn_masks)//2:]

    # Now combine them, while keeping track of the source of each image.  We want a list for val and test corresponding to the source of each image

    val_image_source = []
    val_images = []
    val_masks = []

    for img, mask in zip(cp_val_images, cp_val_masks):
        val_image_source.append('cellpose')
        val_images.append(img)
        val_masks.append(mask)

    for img, mask in zip(bp_val_images, bp_val_masks):
        val_image_source.append('bact_phase')
        val_images.append(img)
        val_masks.append(mask)

    for img, mask in zip(bf_val_images, bf_val_masks):
        val_image_source.append('bact_fluor')
        val_images.append(img)
        val_masks.append(mask)

    for img, mask in zip(tn_val_images, tn_val_masks):
        val_image_source.append('tissuenet')
        val_images.append(img)
        val_masks.append(mask)

    #now shuffle the val_images and val_masks together and the source list
    perm_idx = np.random.permutation(np.arange(len(val_images)))
    val_images = [val_images[i] for i in perm_idx]
    val_masks = [val_masks[i] for i in perm_idx]
    val_image_source = [val_image_source[i] for i in perm_idx]


    # Now let's repeat the same for the test set
    test_image_source = []
    test_images = []
    test_masks = []

    for img, mask in zip(cp_test_images, cp_test_masks):
        test_image_source.append('cellpose')
        test_images.append(img)
        test_masks.append(mask) 

    for img, mask in zip(bp_test_images, bp_test_masks):
        test_image_source.append('bact_phase')
        test_images.append(img)
        test_masks.append(mask)

    for img, mask in zip(bf_test_images, bf_test_masks):
        test_image_source.append('bact_fluor')
        test_images.append(img)
        test_masks.append(mask)

    for img, mask in zip(tn_test_images, tn_test_masks):
        test_image_source.append('tissuenet')
        test_images.append(img)
        test_masks.append(mask)

    # Now shuffle the test_images and test_masks together and the source list
    perm_idx = np.random.permutation(np.arange(len(test_images)))
    test_images = [test_images[i] for i in perm_idx]
    test_masks = [test_masks[i] for i in perm_idx]
    test_image_source = [test_image_source[i] for i in perm_idx]

    # Let's  now pickle dump it


    import pickle

    if to_save:
        # save val set
        with open(os.path.join(save_dir, 'val_set/combo_val_data.pkl'), 'wb') as f:
            pickle.dump({'images': val_images, 'masks': val_masks, 'image_ids': val_image_source}, f, protocol=pickle.HIGHEST_PROTOCOL)
    # save test set
        with open(os.path.join(save_dir, 'test_set/combo_test_data.pkl'), 'wb') as f:
            pickle.dump({'images': test_images, 'masks': test_masks, 'image_ids': test_image_source}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    # Download Data from:
    # Omnipose (bact_phase and bact_fluor): https://osf.io/xmury/
    # Cellpose : https://www.cellpose.org/
    # Tissuenet: : https://datasets.deepcell.org/data
    # Requires data_dir to have the following structure:
    # data/
    #   cellpose/
    #   bact_phase/
    #   bact_fluor/
    #   tissuenetv1.1.npz
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_arguement('--to_save', type=bool, default=False)
    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.to_save)