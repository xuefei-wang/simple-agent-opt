"""
python figs/medsam_analyze_trajectories.py \
    --k=$K \
    --checkpoint_path=$CHECKPOINT_FILE \
    --val_data_path=$VAL_DATA_FILE \
    --test_data_path=$TEST_DATA_FILE \
    --gpu_id=$GPU_ID \
    --json_path=$JSON_PATH # path to preprocessing_func_bank.json
"""
import json
import os
from typing import List, Dict, Callable
import numpy as np
import cv2 as cv
import pandas as pd
import argparse
import sys

_CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.medsam_segmentation import MedSAMTool
from src.data_io import ImageData
from utils.function_bank_utils import should_include_function
from utils.analysis_utils import (
    find_all_metrics, find_highest, find_rolling_highest,
    find_top_k, convert_string_to_function, parse_automl_status,
    extract_metric,
)
import io
import contextlib
import csv
import matplotlib.pyplot as plt

def get_baseline(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='replace') as file:
        baseline_data = file.readlines()
        baseline_dsc = [float(line.split(": ")[1]) for line in baseline_data if "Average DSC metric" in line][0]
        baseline_nsd = [float(line.split(": ")[1]) for line in baseline_data if "Average NSD metric" in line][0]
    return baseline_dsc, baseline_nsd, baseline_dsc + baseline_nsd


def get_new_json(json_path):
    with open(json_path, 'r') as file:
        json_array = json.load(file)

    # Handle json ambiguities
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
        dsc_metric = extract_metric(json_array[i], 'dsc_metric')
        if dsc_metric is None:
            print(f"Error extracting dsc_metric at index {i}")
        data_for_json['dsc_metric'] = dsc_metric

        nsd_metric = extract_metric(json_array[i], 'nsd_metric')
        if nsd_metric is None:
            print(f"Error extracting nsd_metric at index {i}")
        data_for_json['nsd_metric'] = nsd_metric

        new_json.append(data_for_json)
    return new_json

def extract_top_k_preprocessing_functions_to_json(k, json_path, segmenter, test_data_path):
    new_json = get_new_json(json_path)
    top_fns = find_top_k(new_json, lambda x: x['dsc_metric'] + x['nsd_metric'], k)  # Top 3
    print(f"Finished locating the top k = {k} preprocessing & postprocessing functions in the rollout.")

    result_entries = []

    for i, json_dict in enumerate(top_fns):
        print(f"Evaluating test set on the top {i}-th of {len(top_fns)} functions")
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            imgs, boxes, masks = segmenter.loadData(test_data_path)
            spliced_imgs = imgs
            spliced_boxes = boxes
            spliced_masks = masks
            images = ImageData(
                raw=spliced_imgs,
                batch_size=min(8, len(spliced_imgs)),
                image_ids=[i for i in range(len(spliced_imgs))],
                masks=spliced_masks,
                predicted_masks=spliced_masks,
            )
            fn_str_preprocess = json_dict['preprocessing_function']
            fn_str_postprocess = json_dict['postprocessing_function']
            preprocessing_fn = convert_string_to_function(fn_str_preprocess, 'preprocess_images')
            postprocessing_fn = convert_string_to_function(fn_str_postprocess, 'postprocess_preds')
            images = preprocessing_fn(images)
            pred_masks = segmenter.predict(images, spliced_boxes, used_for_baseline=False)
            pred_masks_final = postprocessing_fn(pred_masks)                
            segmenter.evaluate(pred_masks_final, spliced_masks)

        stdout_output = stdout_capture.getvalue()
        agent_dsc, agent_nsd = None, None
        for line in stdout_output.splitlines():
            if "Average DSC metric" in line:
                agent_dsc = float(line.split(": ")[1])
            elif "Average NSD metric" in line:
                agent_nsd = float(line.split(": ")[1])

        combined_test = agent_dsc + agent_nsd if agent_dsc and agent_nsd else None
        combined_val = json_dict['dsc_metric'] + json_dict['nsd_metric'] if json_dict['dsc_metric'] and json_dict['nsd_metric'] else None

        entry = {
            "rank": i + 1,
            "preprocessing_function": fn_str_preprocess,
            "postprocessing_function": fn_str_postprocess,
            "combined_test": combined_test,
            "combined_val": combined_val,
            "automl_status": json_dict['automl_status']
        }
        result_entries.append(entry)

    return result_entries

def plot_scatterplot(output_json_path, baseline_val, baseline_test, scatter_output_path):
    x_vals = []
    y_vals = []
    with open(output_json_path, 'r') as f:
        functions = json.load(f)

    for func in functions:
        val_adv = func['combined_val'] - baseline_val
        test_adv = func['combined_test'] - baseline_test
        x_vals.append(val_adv)
        y_vals.append(test_adv)
    

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals, alpha=0.7)

    plt.xlabel('Advantage over Validation Baseline')
    plt.ylabel('Advantage over Test Baseline')
    plt.title('Scatterplot: Preprocessing Function Improvements')

    # Add reference lines
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)

    # Diagonal reference line: test = val
    grid_size = max(abs(min(x_vals)), abs(max(x_vals)), abs(min(y_vals)), abs(max(y_vals))) + 0.01
    plt.plot([-grid_size, grid_size], [-grid_size, grid_size], 'r--', label='Val = Test')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # Axis limits
    plt.xlim([-grid_size, grid_size])
    plt.ylim([-grid_size, grid_size])

    # Save the figure
    plt.savefig(scatter_output_path)
    print(f"Saved scatterplot to {scatter_output_path}")

def plot_aggregated(json_saved_directories, baselines, k, scatter_output_path):
    """
    Aggregates and plots the top k functions from each timestamp subfolder within the provided directories.
    Points from different directories are colored differently and labeled by their folder names.
    """
    print(json_saved_directories)
    plt.figure(figsize=(10, 10))

    # Color palette
    colors = plt.cm.get_cmap('tab10', len(json_saved_directories))

    for idx, (json_path, (baseline_val, baseline_test)) in enumerate(zip(json_saved_directories, baselines)):
        with open(json_path, 'r') as f:
            functions = json.load(f)

        x_vals = []
        y_vals = []

        for func in functions:
            val_adv = func['combined_val'] - baseline_val
            test_adv = func['combined_test'] - baseline_test
            x_vals.append(val_adv)
            y_vals.append(test_adv)

        folder_name = os.path.basename(os.path.dirname(json_path))
        plt.scatter(x_vals, y_vals, alpha=0.7, label=folder_name, color=colors(idx))

    plt.xlabel('Advantage over Validation Baseline')
    plt.ylabel('Advantage over Test Baseline')
    plt.title(f'Scatterplot: Preprocessing Function Improvements with K={k}')

    # Add reference lines
    plt.axhline(0, color='black', linewidth=1.5)
    plt.axvline(0, color='black', linewidth=1.5)

    # Diagonal reference line: test = val
    all_vals = x_vals + y_vals  # last ones from loop, just to get bounds
    grid_size = max(abs(min(all_vals)), abs(max(all_vals))) + 0.01
    plt.plot([-grid_size, grid_size], [-grid_size, grid_size], 'r--', label='Val = Test')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')

    # Axis limits
    plt.xlim([-grid_size, grid_size])
    plt.ylim([-grid_size, grid_size])

    # Save the figure
    plt.savefig(scatter_output_path)
    print(f"Saved aggregated scatterplot to {scatter_output_path}")

def plot_bar_graph(json_path, baseline_val, baseline_test, bar_output_path):
    plt.figure()
    with open(json_path, 'r') as f:
        functions = json.load(f)
        our_test = functions[0]['combined_test']
    plt.bar('Baseline Test Set', baseline_test, color='b', label=f'Baseline Test Set: {baseline_test:.4f}')
    plt.bar('Our Function Test Set', our_test, color='r', label=f'Our Test Set: {our_test:.4f}')
    
    plt.ylabel('DSC + NSD')
    plt.legend(loc='lower right')
    plt.savefig(bar_output_path)
    print(f"Saved bar graph to {bar_output_path}")

def plot_line_graph(output_path, new_json, combined_metric):
    """ Plot Combined Metric in a single figure """
    combined_metric_lambda = lambda x: x['dsc_metric'] + x['nsd_metric']
    metrics_combined = find_all_metrics(new_json, combined_metric_lambda)
    rolling_highest_combined = find_rolling_highest(new_json, combined_metric_lambda)

    plt.figure(figsize=(10, 5))

    # Plot Combined Metric
    plt.plot(metrics_combined, marker='o', linestyle='-', color='b', label='Eval Combined')
    plt.plot(rolling_highest_combined, marker='x', linestyle='--', color='r', label=f'Rolling Highest Combined: {max(rolling_highest_combined)}')
    plt.axhline(combined_metric, color='g', linestyle='--', label=f'Baseline Combined: {combined_metric}')
    plt.xlabel('Iteration')
    plt.ylabel('Combined Metric')
    plt.title('Combined Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved line graph to {output_path}")

def postprocessing_preds_expert(preds):
    """
    Post-process the low-resolution probability predictions to obtain final segmentation masks.
    """
    import torch.nn.functional as F
    upsampled = F.interpolate(preds, size=(512, 512), mode='bilinear', align_corners=False)
    upsampled = upsampled.squeeze(1)
    seg = (upsampled > 0.5).int()
    return seg

    
def main(json_path, k, modality, gpu_id, val_baseline, test_baseline, test_data_path, checkpoint_path):
    timestamp_path = os.path.dirname(json_path)
    analysis_results_path = os.path.join(timestamp_path, 'analysis_results')
    os.makedirs(analysis_results_path, exist_ok=True)
    top_k_json_output_path = os.path.join(analysis_results_path, 'top_k_functions_results.json')
    top_1_json_output_path = os.path.join(analysis_results_path, 'top_1_functions_results.json')
    bar_output_path = os.path.join(analysis_results_path, 'bar_plot.png')
    line_graph_output_path = os.path.join(analysis_results_path, 'line_graph.png')
    scatter_output_path = os.path.join(analysis_results_path, 'scatter_plot.png')
    

    segmenter = MedSAMTool(gpu_id=gpu_id, checkpoint_path=checkpoint_path)

    results_k = extract_top_k_preprocessing_functions_to_json(k, json_path, segmenter, test_data_path)
    # if file doesn't exist, create  that file
    with open(top_k_json_output_path, 'w') as f:
        json.dump(results_k, f, indent=4)
    
    print("Extracting top k=1 function so we can generate bar graph.")
    results_1 = extract_top_k_preprocessing_functions_to_json(1, json_path, segmenter, test_data_path)
    with open(top_1_json_output_path, 'w') as f:
        json.dump(results_1, f, indent=4)

    plot_line_graph(line_graph_output_path, get_new_json(json_path), val_baseline)
    # plot_scatterplot(top_k_json_output_path, val_baseline, test_baseline, scatter_output_path)
    plot_bar_graph(top_1_json_output_path, val_baseline, test_baseline, bar_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze agent search trajectory.')
    parser.add_argument('--k', type=int, required=True, default=10, help='Number of top functions to run analysis on.')
    parser.add_argument('--checkpoint_path', type=str, required=True, default="", help='Path to the MedSAM checkpoint.')
    parser.add_argument('--val_data_path', type=str, required=True, default="", help='Path to validation dataset.')
    parser.add_argument('--test_data_path', type=str, required=True, default="", help='Path to test dataset.')
    parser.add_argument('--gpu_id', type=int, required=True, default=0, help='GPU ID to use.')
    parser.add_argument('--json_path', type=str, required=False, default="", help='Path to the preprocessing_func_bank.json file.')

    args = parser.parse_args()


    checkpoint_path = args.checkpoint_path
    val_data_path = args.val_data_path
    test_data_path = args.test_data_path
    gpu_id = args.gpu_id
    json_path = args.json_path
                        

    segmenter = MedSAMTool(gpu_id=gpu_id, checkpoint_path=checkpoint_path)

    
    
    imgs, boxes, masks = segmenter.loadData(val_data_path)
    spliced_imgs = imgs
    spliced_boxes = boxes
    spliced_masks = masks
    images = ImageData(
        raw=spliced_imgs,
        batch_size=min(8, len(spliced_imgs)),
        image_ids=[i for i in range(len(spliced_imgs))],
        masks=spliced_masks,
        predicted_masks=spliced_masks,
    )
    pred_masks = segmenter.predict(images, spliced_boxes, used_for_baseline=True, is_rgb=True)
    pred_masks_final = postprocessing_preds_expert(pred_masks)
    metrics_dict = segmenter.evaluate(pred_masks_final, spliced_masks)

    val_baseline = metrics_dict['dsc_metric'] + metrics_dict['nsd_metric']

    imgs, boxes, masks = segmenter.loadData(test_data_path)
    spliced_imgs = imgs
    spliced_boxes = boxes
    spliced_masks = masks
    images = ImageData(
        raw=spliced_imgs,
        batch_size=min(8, len(spliced_imgs)),
        image_ids=[i for i in range(len(spliced_imgs))],
        masks=spliced_masks,
        predicted_masks=spliced_masks,
    )
    pred_masks = segmenter.predict(images, spliced_boxes, used_for_baseline=True, is_rgb=True)
    pred_masks_final = postprocessing_preds_expert(pred_masks)
    metrics_dict = segmenter.evaluate(pred_masks_final, spliced_masks)

    test_baseline = metrics_dict['dsc_metric'] + metrics_dict['nsd_metric']

    metrics_filepath = f"medSAM_segmentation/expert_baseline_performances.json"
    with open(metrics_filepath, "w") as f:
        json_output = {
            "expert_baseline_val_avg_metric": val_baseline,
            "expert_baseline_test_avg_metric": test_baseline,
            "expert_baseline_val_avg_dsc": metrics_dict['dsc_metric'],
            "expert_baseline_val_avg_nsd": metrics_dict['nsd_metric'],
            "expert_baseline_test_avg_dsc": metrics_dict['dsc_metric'],
            "expert_baseline_test_avg_nsd": metrics_dict['nsd_metric'],
        }
        json.dump(json_output, f)


    main(json_path, k=args.k, modality="dermoscopy", gpu_id=gpu_id, val_baseline=val_baseline, test_baseline=test_baseline, test_data_path=test_data_path, checkpoint_path=checkpoint_path)
