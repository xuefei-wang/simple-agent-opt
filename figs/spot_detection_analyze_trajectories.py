"""
CUDA_VISIBLE_DEVICES=$GPU_ID python figs/spot_detection_analyze_trajectories.py \
    --k=$K \
    --data_path=$DATA_PATH \
    --json_path=$JSON_PATH # path to preprocessing_func_bank.json
"""
import os
import sys
import json
import numpy as np
from typing import List, Dict, Callable
import argparse
import matplotlib.pyplot as plt
import cv2 as cv
import logging
import sys

from typing import Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import glob

# Dynamically add the project root to sys.path
# This allows Python to find the 'src' module correctly
_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.spot_detection import DeepcellSpotsDetector
from src.data_io import ImageData
from utils.function_bank_utils import should_include_function
from utils.analysis_utils import (
    find_all_metrics, find_highest, find_rolling_highest,
    find_top_k, convert_string_to_function, parse_automl_status,
    extract_metric,
)

from skimage.feature import peak_local_max


def postprocess_preds_expert(preds):
    """Convert raw prediction to a predicted point list using
    ``skimage.feature.peak_local_max`` to determine local maxima in classification
    prediction image, and their corresponding regression values will be used to
    create a final spot position prediction which will be added to the output spot
    center coordinates list.

    Args:
        preds (array): a dictionary of predictions with keys `'classification'` and
            `'offset_regression'` 

    Returns:
        array: spot center coordinates of the format [[y0, x0], [y1, x1],...]
    """
    import numpy as np
    from skimage.feature import peak_local_max
    dot_centers = []
    for ind in range(np.shape(preds['classification'])[0]):
        dot_pixel_inds = peak_local_max(preds['classification'][ind, ..., 1],
                                        min_distance=2,
                                        threshold_abs=0.98)

        delta_y = preds['offset_regression'][ind, ..., 0]
        delta_x = preds['offset_regression'][ind, ..., 1]
        dot_centers.append(np.array(
            [[y_ind + delta_y[y_ind, x_ind],
            x_ind + delta_x[y_ind, x_ind]] for y_ind, x_ind in dot_pixel_inds]))

    return dot_centers

def main(json_path: str, data_path: str, output_dir: str, k):
    # Let's save these results into a new subfolder of output_dir
    output_dir = os.path.join(output_dir, 'analysis_results')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as file:
        json_array = json.load(file)
        
    # --- Initialize spot detector tool ---
    deepcell_spot_detector = DeepcellSpotsDetector()
    spots_data = np.load(f"{data_path}/val.npz", allow_pickle=True)

    # --- Prepare ImageData ---
    batch_size = spots_data['X'].shape[0]
    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])
    pred = deepcell_spot_detector.predict(images)
    pred_final = postprocess_preds_expert(pred)

    metrics_val = deepcell_spot_detector.evaluate(pred_final, spots_data['y'])


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
        avg_prec = extract_metric(json_array[i], 'f1_score')
        if avg_prec is None:
            print(f"Error extracting f1_score at index {i}")
        data_for_json['f1_score'] = avg_prec
        new_json.append(data_for_json) 


    # Plot results for val set
    json_array = new_json
    metric_lambda = lambda obj: obj['f1_score']

    metrics = find_all_metrics(json_array, metric_lambda)
    highest_metric_obj   = find_highest(json_array, metric_lambda)
    rolling_highest = find_rolling_highest(json_array, metric_lambda)

    plt.figure()
    plt.plot(metrics, marker='o', linestyle='-', color='b', label='Eval metric')
    plt.plot(rolling_highest, marker='x', linestyle='--', color='r', label=f'Rolling highest metric: {max(rolling_highest)}')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title(f'Validation Metrics during Agent Search: {json_path.split("/")[-2]}')

    plt.axhline(metrics_val['f1_score'], color='cyan', linestyle='--', label=f"Baseline val F1: {metrics_val['f1_score']}") 
    plt.legend(loc='best')

    plt.savefig(os.path.join(output_dir, 'metrics_during_agent_search.png'))

    # Now begin the image analysis and attempt to convert string to function

    best_preprocessing_function = highest_metric_obj['preprocessing_function']
    best_postprocessing_function = highest_metric_obj['postprocessing_function']
    best_preprocessing_function = convert_string_to_function(best_preprocessing_function, 'preprocess_images')
    best_postprocessing_function = convert_string_to_function(best_postprocessing_function, 'postprocess_preds')

    # Print a dump of the function to a text file
    with open(os.path.join(output_dir, 'best_function.txt'), 'w') as file:
        file.write(highest_metric_obj['preprocessing_function'])
        file.write('\n\n')
        file.write(highest_metric_obj['postprocessing_function'])
    

    # Baseline on test
    test_path = f'{data_path}/test.npz'

    spots_data = np.load(f"{test_path}", allow_pickle=True)
    batch_size = spots_data['X'].shape[0]
    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])
    pred = deepcell_spot_detector.predict(images)
    pred_final = postprocess_preds_expert(pred)
    metrics_test_baseline = deepcell_spot_detector.evaluate(pred_final, spots_data['y'])


    # Agent on test    
    spots_data = np.load(f"{test_path}", allow_pickle=True)

    # --- Prepare ImageData ---
    batch_size = spots_data['X'].shape[0]
    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])    

    best_preprocessed_images = best_preprocessing_function(images)
    pred_test = deepcell_spot_detector.predict(best_preprocessed_images)
    pred_final_test = best_postprocessing_function(pred_test)
    metrics_test_our_function = deepcell_spot_detector.evaluate(pred_final_test, spots_data['y'])



    plt.figure()
    plt.bar('baseline', metrics_test_baseline['f1_score'], color='b', label=f'Baseline Test Set: {metrics_test_baseline["f1_score"]}')
    plt.bar('agent designed', metrics_test_our_function['f1_score'], color='r', label=f'Our function Test Set: {metrics_test_our_function["f1_score"]}')
    plt.xlabel('Method')
    plt.ylabel('Average F1')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'test_set_metrics.png'))


    # Save the best preprocessing function to a text file
    best_preprocessing_function_str = highest_metric_obj['preprocessing_function']
    best_postprocessing_function_str = highest_metric_obj['postprocessing_function']
    best_preprocessing_function = convert_string_to_function(best_preprocessing_function_str, 'preprocess_images')
    best_postprocessing_function = convert_string_to_function(best_postprocessing_function_str, 'postprocess_preds')

    # Print a dump of the function to a text file
    with open(os.path.join(output_dir, 'best_function.txt'), 'w') as file:
        file.write(highest_metric_obj['preprocessing_function'])
        file.write('\n\n')
        file.write(highest_metric_obj['postprocessing_function'])
        
    # Save the baseline metrics to a text file
    expert_baseline_performances = {
        "expert_baseline_val_f1": metrics_val['f1_score'],
        "expert_baseline_test_f1": metrics_test_baseline['f1_score']
    }
    with open(os.path.join(output_dir, 'expert_baseline_performances.json'), 'w') as file:
        json.dump(expert_baseline_performances, file, indent=4)
    print(f"Saved expert baseline performances to {os.path.join(output_dir, 'expert_baseline_performances.json')}")

    ## Evaluate TOP K
    top_k_functions = find_top_k(json_array, metric_lambda, k)

    # Now let's evaluate the top k functions on the test set
    top_k_functions_results_val = []
    top_k_functions_results_test = []
    top_k_functions_str = []
    top_k_functions_automl_status = []

    for function_item in top_k_functions:
        current_function_str = (function_item['preprocessing_function'], function_item['postprocessing_function'])
        current_metrics_val_float = function_item['f1_score']
        current_automl_status = function_item['automl_status']

        if current_function_str == (best_preprocessing_function_str, best_postprocessing_function_str): # Compare strings to avoid issues with function object comparison
            current_metrics_test_dict = metrics_test_our_function # Use already computed result for the best function
            function_str_to_save = current_function_str
        else:
            cur_preprocessing_fn_obj = convert_string_to_function(current_function_str[0], 'preprocess_images')
            cur_postprocessing_fn_obj = convert_string_to_function(current_function_str[1], 'postprocess_preds')
            
            # Ensure fresh ImageData object for each preprocessing
            batch_size = spots_data['X'].shape[0]
            images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])

            cur_preprocessed_images_test = cur_preprocessing_fn_obj(images)
            # Evaluate with non_privileged_segmenter, same as the best agent function
            pred_indiv_test = deepcell_spot_detector.predict(cur_preprocessed_images_test)
            pred_indiv_test_final = cur_postprocessing_fn_obj(pred_indiv_test)
            current_metrics_test_dict = deepcell_spot_detector.evaluate(pred_indiv_test_final, spots_data['y'])
            function_str_to_save = current_function_str
        
        top_k_functions_results_test.append(current_metrics_test_dict)
        top_k_functions_results_val.append(current_metrics_val_float)
        top_k_functions_str.append(function_str_to_save)
        top_k_functions_automl_status.append(current_automl_status)

    print(f"Evaluated top {k} functions on test set.")

    # Save the results into a new file (json)
    top_k_functions_results_output = []
    for i in range(len(top_k_functions)):
        rank = i + 1
        preprocessing_function_string = top_k_functions_str[i][0]
        postprocessing_function_string = top_k_functions_str[i][1]
        test_metrics_dict = top_k_functions_results_test[i]
        val_metric_float = top_k_functions_results_val[i]

        top_k_functions_results_output.append({
            "rank": rank,
            "preprocessing_function": preprocessing_function_string,
            "postprocessing_function": postprocessing_function_string,
            "average_f1_test": test_metrics_dict['f1_score'],
            "average_f1_val": val_metric_float,
            "automl_status": top_k_functions_automl_status[i],
        })
    
    with open(os.path.join(output_dir, 'top_k_functions_results.json'), 'w') as file:
        json.dump(top_k_functions_results_output, file, indent=4)
    print(f"Saved top-k function results to {os.path.join(output_dir, 'top_k_functions_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--k', type=int, required=True, default=10, help='Number of top functions to run analysis on.')
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory that contains the val and test npz files."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file of preprocessing function bank."
    )
    args = parser.parse_args()
    
    data_path = args.data_path
    json_path = args.json_path
    
    main(json_path, data_path, os.path.dirname(json_path), args.k)

    


