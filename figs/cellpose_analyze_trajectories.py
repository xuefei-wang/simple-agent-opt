"""
python figs/cellpose_analyze_trajectories.py \
    --k=$K \
    --data_path=$DATA_FOLDER \
    --gpu_id=$GPU_ID \
    --json_path=$JSON_PATH # path to preprocessing_func_bank.json
"""
import os 
import sys
import json
import numpy as np
from typing import List, Dict, Callable
import argparse
import matplotlib.pyplot as plt
# from src.cellpose_segmentation import CellposeTool
# from src.data_io import ImageData
import cv2 as cv
import logging
import sys
import pickle

from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import glob
import numpy as np
import cv2 as cv

_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Project-specific imports from 'src'
from src.cellpose_segmentation import CellposeTool
from src.data_io import ImageData
# from src.tools import BaseSegmenter # Added direct import
from src.utils import set_gpu_device # Added direct import
from utils.function_bank_utils import should_include_function
from utils.analysis_utils import (
    find_all_metrics, find_highest, find_rolling_highest,
    find_top_k, convert_string_to_function, parse_automl_status,
    extract_metric,
)

from cellpose import models, denoise
from cellpose.io import imread
from cellpose.metrics import average_precision, mask_ious, boundary_scores, aggregated_jaccard_index, flow_error

import os


class PriviligedCellposeTool():
    """
    PriviligedCellposeTool is a class that provides a simple interface for the Cellpose model. 
    """
    def __init__(self, model_name: str = "cyto3", device: int = 0, channels: List[int] = [2,1], to_normalize: bool = False, model_kwargs: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.to_normalize = to_normalize

        if device == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        # R: cytoplasm, G: nucleus
        self.channels = channels

        assert self.model_name in ["cyto3", "denoise_cyto3"], f"Model name {self.model_name} not recognized"

        if self.model_name == "cyto3":
            self.segmenter = models.Cellpose(model_type='cyto3',device=self.device, gpu=True)
        elif self.model_name == "denoise_cyto3":
            self.segmenter = denoise.CellposeDenoiseModel(model_type='cyto3', restore_type='denoise_cyto3', device=self.device, gpu=True)

    def predict(self, images: ImageData, batch_size: int = 8) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[np.ndarray], Any]:
        """
        Predict masks for a batch of images. 
        Args:
            images: ImageData object containing a batch of images. Contains 'raw' and 'masks' attributes in the format of standard ImageData object 
            [batch_size, height, width, channels]. Images provided must be in the format of standard ImageData object and must have two channels, the first channel being the cytoplasm and the second channel being the nucleus.
            batch_size: batch size for prediction
        Returns: 
            masks (List[np.ndarray]): List of labelled images (numpy arrays), where 0=no masks; 1,2,...=mask labels for all pixels in the image
        """

        # Old returns:
        # flows (List[List[np.ndarray]]): List of flow outputs per image:
        # flows[k][0] = XY flow in HSV 0-255
        # flows[k][1] = XY(Z) flows at each pixel
        # flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics)
        # flows[k][3] = final pixel locations after Euler integration
        # styles (List[np.ndarray]): List of style vectors (size 256) summarizing each image
        # extra (Any): Diameters if using Cellpose model, input images if using denoise model
        to_normalize = self.to_normalize
        raw_list=images.raw
        if self.model_name == "cyto3":
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
        elif self.model_name == "denoise_cyto3":
            masks, flows, styles, extra = self.segmenter.eval(raw_list, diameter=None, channels=self.channels, normalize=to_normalize, batch_size=batch_size)
              
        return masks#, flows, styles, extra
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], precision_index: int = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the performance of the model.
        Args:
            pred_masks: predicted masks
            gt_masks: ground truth masks
        Returns:
            metrics: dictionary of metrics. Contains average_precision at IoU thresholds [0.5, 0.75, 0.9]
            losses: dictionary of losses. Contains bce_loss
        """
        metrics = {}
        losses = {}
        ap, tp, fp, fn  = average_precision(pred_masks, gt_masks)
        metrics["average_precision"] = np.nanmean(ap, axis=0) # Average over all images
        
        spatial_shape = pred_masks[0].shape[0:2]
        for x in pred_masks:
            if x.shape[0:2] != spatial_shape:
                different_spatial_shapes = True
                break
            else:
                different_spatial_shapes = False

        if not different_spatial_shapes:
            criterion2 = nn.BCEWithLogitsLoss(reduction="mean")
            loss2 = criterion2(torch.Tensor(np.array(pred_masks)), torch.from_numpy(np.array(gt_masks )> 0.5).squeeze().float())
            losses["bce_loss"] = loss2.item()#/len(pred_masks)
        else:
            criterion2 = nn.BCEWithLogitsLoss(reduction="none")
            total_loss = 0
            for pred, gt in zip(pred_masks, gt_masks):
                # Make sure both are tensors with matching dimensions
                pred_tensor = torch.Tensor(pred).view(-1)  # Flatten
                gt_tensor = torch.from_numpy(gt > 0.5).float().view(-1)  # Flatten

                # Compute loss for this pair
                mask_loss = criterion2(pred_tensor, gt_tensor).mean()
                total_loss += mask_loss.item()

            losses["bce_loss"] = total_loss / len(pred_masks)
        
        # Let's simplify and only return average precision at [0.5]
        return {"average_precision": metrics["average_precision"][precision_index].item()}

       # return metrics, losses

    def preprocess(self, image_data: ImageData) -> ImageData:
        """We don't need to preprocess the images for Cellpose"""
        return image_data

    def loadData(self, data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load the data from the data path and return a tuple of lists of raw images and gt masks, each as numpy arrays"""
        max_val = 65535 # 16-bit images
        files = sorted(glob.glob(data_path + '*_img.png'))
        raw_images = [(imread(f)).astype(np.float32)/max_val for f in files]
        gt_masks = [imread(f.split('.')[0][:-3] + 'masks' + '.' + f.split('.')[1]) for f in files]
        gt_masks = [np.expand_dims(mask, axis=2) for mask in gt_masks]
        return raw_images, gt_masks
    
    def loadCombinedDataset(self, data_path: str, dataset_size: int = 256) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """Used with combined datasets."""
        # Load all images and masks
        file = glob.glob(os.path.join(data_path, '*'))
        with open(file[0], 'rb') as f:
            data = pickle.load(f)
        images = data['images'][:dataset_size]
        masks = data['masks'][:dataset_size]
        image_ids = data['image_ids'][:dataset_size]
        return images, masks, image_ids 
    
    def evaluateDisaggregated(self, imageData_obj: ImageData, avg_precision_idx: int = 0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate the performance of the model on a disaggregated dataset"""
        metrics = {}
        losses = {}
        pred_masks = imageData_obj.predicted_masks
        gt_masks = imageData_obj.masks
        ap, tp, fp, fn  = average_precision(pred_masks, gt_masks)

        img_source_ids = np.array(imageData_obj.image_ids) 
        metrics = {'average_precision': np.nanmean(ap, axis=0)[0].item()}

        bool_mask = img_source_ids == 'cellpose'
        cp_only_ap = ap[bool_mask]  

        bool_mask = img_source_ids == 'bact_phase'
        bp_only_ap = ap[bool_mask]  

        bool_mask = img_source_ids == 'bact_fluor'
        bf_only_ap = ap[bool_mask]  

        bool_mask = img_source_ids == 'tissuenet'
        tn_only_ap = ap[bool_mask]  


        per_dataset = {
            'cellpose': np.nanmean(cp_only_ap, axis=0)[avg_precision_idx].item(),
            'bact_phase': np.nanmean(bp_only_ap, axis=0)[avg_precision_idx].item(),
            'bact_fluor': np.nanmean(bf_only_ap, axis=0)[avg_precision_idx].item(),
            'tissuenet': np.nanmean(tn_only_ap, axis=0)[avg_precision_idx].item()
        }
        
        metrics['disaggregated_average_precision'] = {}
        for name, data in per_dataset.items():
            mean_result = np.nanmean(data, axis=0)
            value = mean_result 
            metrics['disaggregated_average_precision'][name] = None if np.isnan(value) else float(value)

        return metrics
    
def postprocessing_preds_expert(preds):
    """ Fills holes in preds (2D/3D) and discards preds smaller than min_size.

    This function fills holes in each mask using scipy.ndimage.morphology.binary_fill_holes.
    It also removes preds that are smaller than the specified min_size.

    Parameters:
    preds (list[ndarray]): Int, 2D or 3D array of labelled masks.
        0 represents no mask, while positive integers represent mask labels.
        The size can be [Ly x Lx] or [Lz x Ly x Lx].

    Returns:
        list[ndarray]: Int, 2D or 3D array of masks with holes filled and small preds removed.
            0 represents no mask, while positive integers represent mask labels.
            The size is [Ly x Lx] or [Lz x Ly x Lx].
    """
    from scipy.ndimage import find_objects, binary_fill_holes
    
    min_size = 15
    
    new_preds = []
    for mask in preds:
        slices = find_objects(mask)
        j = 0
        for i, slc in enumerate(slices):
            if slc is not None:
                msk = mask[slc] == (i + 1)
                npix = msk.sum()
                if min_size > 0 and npix < min_size:
                    mask[slc][msk] = 0
                elif npix > 0:
                    if msk.ndim == 3:
                        for k in range(msk.shape[0]):
                            msk[k] = binary_fill_holes(msk[k])
                    else:
                        msk = binary_fill_holes(msk)
                    mask[slc][msk] = (j + 1)
                    j += 1
        new_preds.append(mask)
    return new_preds


def main(json_path: str, data_path: str, output_dir: str, precision_index: int = 0, device: int = 0, dataset_size: int = 100, batch_size: int = 16, k: int = 10):
    # Let's save these results into a new subfolder of output_dir
    output_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as file:
        json_array = json.load(file)


    # ---  Expert baseline (val) --- 

    # Use priviliged CellposeTool to get "best" results, to_normalize=True
    priviliged_segmenter = PriviligedCellposeTool(model_name="cyto3", device=device, channels=[2,1], to_normalize=True)

    # raw_images_val, gt_masks_val = priviliged_segmenter.loadData(os.path.join(data_path, 'val_set/'))
    raw_images_val, gt_masks_val, image_sources_val = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'val_set/'), dataset_size=dataset_size)

    images_val = ImageData(raw=raw_images_val, masks=gt_masks_val, batch_size=batch_size, image_ids=image_sources_val)
    preds = priviliged_segmenter.predict(images_val, batch_size=images_val.batch_size)
    final_preds = postprocessing_preds_expert(preds)
    images_val.predicted_masks = final_preds
    metrics_val_expert_baseline = priviliged_segmenter.evaluateDisaggregated(images_val)
    print('Evaluated expert baseline on val (cyan line)')

    # --- Expert baseline (test) ---
    test_dataset_size = 808 # hardcoded  --- 

    raw_images_test, gt_masks_test, image_sources_test = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)

    images_test = ImageData(raw=raw_images_test, masks=gt_masks_test, batch_size=batch_size, image_ids=image_sources_test)
    preds = priviliged_segmenter.predict(images_test, batch_size=images_test.batch_size)
    final_preds = postprocessing_preds_expert(preds)
    images_test.predicted_masks = final_preds
    metrics_test_expert_baseline = priviliged_segmenter.evaluateDisaggregated(images_test)
    print("Evaluated expert baseline on test (blue bar)")


    # --- Save or Load expert baseline ---
    save_path = os.path.join(output_dir, 'expert_baseline_performances.json')
    
    # Save expert baseline performances to a JSON file
    expert_baseline_performances = {
        "expert_baseline_val_avg_precision": metrics_val_expert_baseline['average_precision'],
        "expert_baseline_test_avg_precision": metrics_test_expert_baseline['average_precision'],
        "disaggregated_expert_baseline_test_avg_precision": metrics_test_expert_baseline['disaggregated_average_precision'],
        'disaggregated_expert_baseline_val_avg_precision': metrics_val_expert_baseline['disaggregated_average_precision']
    }

    with open(save_path, 'w') as file:
        json.dump(expert_baseline_performances, file, indent=4)
    print(f"Saved expert baseline performances to {save_path}")


    
    # handle json ambiguities
    new_json = []
    for i in range(len(json_array)):
        data_for_json = {
            'preprocessing_function': json_array[i]['preprocessing_function'],
            'postprocessing_function': json_array[i]['postprocessing_function'],
            'automl_status': parse_automl_status(json_array[i]),
        }
        # Preserve original automl flags for should_include_function()
        if 'automl_optimized' in json_array[i]:
            data_for_json['automl_optimized'] = json_array[i]['automl_optimized']
        if 'automl_superseded' in json_array[i]:
            data_for_json['automl_superseded'] = json_array[i]['automl_superseded']
        avg_prec = extract_metric(json_array[i], 'average_precision')
        if avg_prec is None:
            print(f"Error extracting average_precision at index {i}")
        data_for_json['average_precision'] = avg_prec
        data_for_json['disaggregated_average_precision'] = json_array[i]['overall_metrics']['disaggregated_average_precision']
        new_json.append(data_for_json) 


    # Plot results for val set
    json_array = new_json
    metric_lambda = lambda obj: obj['average_precision']

    metrics = find_all_metrics(json_array, metric_lambda)
    highest_metric_obj   = find_highest(json_array, metric_lambda)
    rolling_highest = find_rolling_highest(json_array, metric_lambda)

    plt.figure()
    plt.plot(metrics, marker='o', linestyle='-', color='b', label='Eval metric')
    plt.plot(rolling_highest, marker='x', linestyle='--', color='r', label=f'Rolling highest metric: {max(rolling_highest)}')
    plt.xlabel('Iteration')
    plt.ylabel('Average Precision @ IoU=0.5')
    plt.title(f'Validation Metrics during Agent Search: {json_path.split("/")[-2]}')

    plt.axhline(metrics_val_expert_baseline['average_precision'], color='cyan', linestyle='--', label=f"Baseline: cellpose's min-max norm {metrics_val_expert_baseline['average_precision']}") 
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, 'metrics_during_agent_search.png'))

    # Now begin the image analysis and attempt to convert string to function

    best_preprocessing_function_str = highest_metric_obj['preprocessing_function']
    best_postprocessing_function_str = highest_metric_obj['postprocessing_function']
    best_preprocessing_function = convert_string_to_function(best_preprocessing_function_str, 'preprocess_images')
    best_postprocessing_function = convert_string_to_function(best_postprocessing_function_str, 'postprocess_preds')

    # Print a dump of the function to a text file
    with open(os.path.join(output_dir, 'best_functions.txt'), 'w') as file:
        file.write(highest_metric_obj['preprocessing_function'])
        file.write('\n\n')
        file.write(highest_metric_obj['postprocessing_function'])


    # Agent's best function on test set, without to_normalize
    non_privileged_segmenter_test = CellposeTool(model_name="cyto3", device=device) # ensure fresh segmenter for test
    raw_images_test_agent, gt_masks_test_agent, image_sources_test_agent = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)

    images_test_agent = ImageData(raw=raw_images_test_agent, masks=gt_masks_test_agent, batch_size=batch_size, image_ids=image_sources_test_agent)
    best_preprocessed_images_test = best_preprocessing_function(images_test_agent)
    # non_privileged_segmenter_test is already initialized
    preds = non_privileged_segmenter_test.predict(best_preprocessed_images_test, batch_size=batch_size)
    final_preds = best_postprocessing_function(preds)
    images_test_agent.predicted_masks = preds
    metrics_test_agent_function = non_privileged_segmenter_test.evaluateDisaggregated(images_test_agent)
    print("Evaluated top performing agent function on test (red bar)")


    plt.figure()
    plt.bar('expert_baseline', metrics_test_expert_baseline['average_precision'], color='b', label=f"Expert Baseline (min-max norm) Test Set: {metrics_test_expert_baseline['average_precision']:.4f}")
    plt.bar('agent_designed', metrics_test_agent_function['average_precision'], color='r', label=f"Agent Designed Test Set: {metrics_test_agent_function['average_precision']:.4f}")
    plt.xlabel('Method')
    plt.ylabel('Average Precision')
    plt.legend(loc='lower right')
    plt.title(f'Test Set Performance Comparison: {json_path.split("/")[-2]}')
    plt.savefig(os.path.join(output_dir, 'test_set_metrics.png'))


    # Let's find the top performing k functions, evaluate them on the test set and save the results into a new file (pickle)
    top_k_functions = find_top_k(json_array, metric_lambda, k)

    # Now let's evaluate the top k functions on the test set
    top_k_functions_results_val = []
    top_k_functions_results_test = []
    top_k_functions_str = []
    top_k_functions_disaggregated_val = []
    top_k_functions_disaggregated_test = []
    top_k_functions_automl_status = []

    # priviliged_segmenter_for_top_k = PriviligedCellposeTool(model_name="cyto3", device=device, channels=[2,1], to_normalize=True)
    # Use non_privileged_segmenter_test for evaluating agent functions, as per earlier logic for "Our function, without to_normalize"
    segmenter_for_top_k_agent_eval = CellposeTool(model_name="cyto3", device=device)


    for idx, function_item in enumerate(top_k_functions):
        print(f"Evaluating {idx+1} of top {len(top_k_functions)}")
        current_function_str = (function_item['preprocessing_function'], function_item['postprocessing_function'])
        current_metrics_val_float = function_item['average_precision']
        current_metrics_val_dict = function_item['disaggregated_average_precision']
        current_automl_status = function_item['automl_status']
        if current_function_str == (best_preprocessing_function_str, best_postprocessing_function_str): # Compare strings to avoid issues with function object comparison
            current_metrics_test_dict = metrics_test_agent_function # Use already computed result for the best function
            function_str_to_save = current_function_str
        else:
            cur_preprocessing_fn_obj = convert_string_to_function(current_function_str[0], 'preprocess_images')
            cur_postprocessing_fn_obj = convert_string_to_function(current_function_str[1], 'postprocess_preds')
            
            # Ensure fresh ImageData object for each preprocessing
            raw_images_test, gt_masks_test, image_sources_test = priviliged_segmenter.loadCombinedDataset(os.path.join(data_path, 'test_set/'), dataset_size=test_dataset_size)
            images_test = ImageData(raw=raw_images_test, masks=gt_masks_test, batch_size=batch_size, image_ids=image_sources_test)
            current_images_to_preprocess_test = images_test

            cur_preprocessed_images_test = cur_preprocessing_fn_obj(current_images_to_preprocess_test)
            # Evaluate with non_privileged_segmenter, same as the best agent function
            preds = segmenter_for_top_k_agent_eval.predict(cur_preprocessed_images_test, batch_size=batch_size)
            final_preds = cur_postprocessing_fn_obj(preds)
            images_test.predicted_masks = final_preds
            current_metrics_test_dict = segmenter_for_top_k_agent_eval.evaluateDisaggregated(images_test)
            function_str_to_save = current_function_str
        
        top_k_functions_results_test.append(current_metrics_test_dict)
        top_k_functions_results_val.append(current_metrics_val_float)
        top_k_functions_str.append(function_str_to_save)
        top_k_functions_disaggregated_test.append(current_metrics_test_dict['disaggregated_average_precision'])
        top_k_functions_disaggregated_val.append(current_metrics_val_dict)
        top_k_functions_automl_status.append(current_automl_status)

    print(f"Evaluated top {k} functions on test set using non-privileged segmenter.")

    # Save the results into a new file (json)
    top_k_functions_results_output = []
    for i in range(len(top_k_functions)):
        rank = i + 1
        preprocessing_function_string = top_k_functions_str[i][0]
        postprocessing_function_string = top_k_functions_str[i][1]
        test_metrics_dict = top_k_functions_results_test[i]
        val_metric_float = top_k_functions_results_val[i]
        val_metrics_dict = top_k_functions_results_val[i]
        test_metrics_dict = top_k_functions_results_test[i]
        top_k_functions_results_output.append({
            "rank": rank,
            "preprocessing_function": preprocessing_function_string,
            "postprocessing_function": postprocessing_function_string,
            "average_precision_test": test_metrics_dict['average_precision'],
            "average_precision_val": val_metric_float,
            "disaggregated_average_precision_test": top_k_functions_disaggregated_test[i],
            "disaggregated_average_precision_val": top_k_functions_disaggregated_val[i],
            "automl_status": top_k_functions_automl_status[i]
        })
    
    # Let's convert nans to None for all top_k_functions_results_outputs.  the only possible ones are disaggregated_average_precision_test and disaggregated_average_precision_val
    for k_func_obj in top_k_functions_results_output:
        for key, value in k_func_obj.items():
            if key == 'disaggregated_average_precision_test' or key == 'disaggregated_average_precision_val':
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float) and np.isnan(sub_value):
                        k_func_obj[key][sub_key] = None
    
    
    with open(os.path.join(output_dir, 'top_k_functions_results.json'), 'w') as file:
        json.dump(top_k_functions_results_output, file, indent=4)
    print(f"Saved top-k function results to {os.path.join(output_dir, 'top_k_functions_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--k', type=int, required=True, default=10, help='Number of top functions to run analysis on.')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the data directory.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use. Use -1 for CPU.')
    args = parser.parse_args()

    data_path = args.data_path
    gpu_id = args.gpu_id
    json_path = args.json_path

    print(json_path)
    output_dir = os.path.dirname(json_path)
    main(json_path, data_path, output_dir, precision_index=0, device=gpu_id, dataset_size=100, batch_size=16, k=args.k)