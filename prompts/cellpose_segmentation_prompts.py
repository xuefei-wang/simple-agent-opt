from prompts.task_prompts import TaskPrompts

class CellposeSegmentationPromptsWithSkeleton(TaskPrompts):
    """Task prompts for cell segmentation analysis. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown
    This is a three-channel image dataset for biological segmentation, consisting of images from different experiments and different settings - a heterogenous dataset of many different object types.  There is a particular focus on biological microscopy images, including cells, sometimes with nuclei labeled in a separate channel.
    The images have pixel values between 0 and 1 and are in float32 format.
    Channel[0] is the nucleus, channel[1] is the cytoplasm, and channel[2] is empty, however not all images have any nuclear data.
    We want to increase the neural network tool's performance at segmenting cells with cell perimeter masks that have high Intersection over Union (IoU) with the ground truth masks.
    The cell images have dimensions (B, L, W, C) = (batch, length, width, channel). To correctly predict masks, the images provided must be in the format of standard ImageData object and must maintain channel dimensions and ordering.
    """

    postprocessing_skeleton_filename = "cellpose_segmentation_expert_postprocessing_skeleton.py.txt"

    def get_task_details(self):
        return f"""
    All of you should work together to write {self.k_word} pairs of preprocessing postprocessing functions that improve segmentation performance.
    We provided APIs for both preprocessing and postprocessing functions. You should use functions from useful libraries including but not limited to OpenCV, NumPy, Skimage, Scipy, to implement novel and effective functions.
    It might make sense to start the process with small preprocessing functions, and then build up to more complex functions depending on the performance of the previous functions.

    1. Based on previous preprocessing functions and their performance (provided below), suggest {self.k_word} new unique preprocessing & postprocessing function pairs.
    2. For preprocessing, the images after preprocessing must still conform to the format specified in the ImageData API. Maintenance of channel identity is critical and channels should not be merged. For postprocessing, it is also critical to maintain the output format as the sample function provided.
    3. The environment will handle all data loading, evaluation, and logging of the results.  Your only job is to write the preprocessing and postprocessing function pairs.
    4. For this task, if all {self.k_word} function pairs are evaluated correctly, only one iteration is allowed, even if the performance is not satisfactory.
    5. Do not terminate the conversation until the new functions are evaluated and the numerical performance metrics are logged.
    6. Extremely important: Do not terminate the conversation until each of the {self.k_word} new function pairs are evaluated AND their results are written to the function bank.
    7. Recall, this is a STATELESS kernel, so all functions, imports, etc. must be provided in the script to be executed. Any history between previous iterations exists solely as provided preprocessing functions and their performance metrics.
    8. Do not write any code outside of the preprocessing and postprocessing functions.
    9. Do not modify the masks under any circumstances.
    """

    def get_pipeline_metrics_info(self):
        return f"""
    The following metrics are used to evaluate the performance of the pipeline: average_precision.
    The average_precision is the average precision score of the pipeline at an Intersection over Union (IoU) threshold of 0.5.
    """

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word, dataset_size=256, batch_size=16):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            function_bank_path=function_bank_path,
            dataset_size=dataset_size,
            batch_size=batch_size,
            k=k,
            k_word=k_word,
        )

    def get_template_replacements(self) -> dict:
        return {
            "task_extra_imports": "",
            "task_imports": "    from src.cellpose_segmentation import CellposeTool",
            "task_extra_config": "dataset_size = {dataset_size}\nbatch_size = {batch_size}",
            "task_setup": (
                '    logger.info("Initializing CellposeTool...")\n'
                '    segmenter = CellposeTool(model_name="cyto3", device=gpu_id)\n'
                '    logger.info(f"Loading data from {data_path}...")\n'
                '    raw_images, gt_masks, image_sources = segmenter.loadCombinedDataset(data_path, dataset_size)\n'
                '    logger.info(f"Data loaded: {len(raw_images)} images.")\n'
                '    logger.info(f"[Single Image] Loading single image as a tester...")\n'
                '\n'
                '    # --- Prepare ImageData ---\n'
                '    images = ImageData(raw=raw_images, batch_size=batch_size, image_ids=image_sources)\n'
                '    single_image_data = ImageData(raw=raw_images[:2], batch_size=2, masks=gt_masks[:2], image_ids=image_sources[:2])'
            ),
            "task_predict_single": "segmenter.predict(single_image_data, batch_size=single_image_data.batch_size)",
            "task_postprocess_check_single": "",
            "task_evaluate_single": (
                "        cur_single_image_data = ImageData(raw=raw_images[:2], batch_size=2, masks=gt_masks[:2], image_ids=image_sources[:2], predicted_masks=pred)\n"
                "        k_overall_metrics_single_img.append(segmenter.evaluateDisaggregated(cur_single_image_data))"
            ),
            "task_predict_full": "segmenter.predict(preprocessed_images, batch_size=preprocessed_images.batch_size)",
            "task_postprocess_check_full": "",
            "task_evaluate_full": (
                "        new_images = ImageData(raw=raw_images, batch_size=batch_size, image_ids=image_sources, masks=gt_masks, predicted_masks=pred)\n"
                "        k_overall_metrics.append(segmenter.evaluateDisaggregated(new_images))"
            ),
            "task_metrics_log": (
                "    if k_overall_metrics:\n"
                "        avg_precision_values = [metric.get('average_precision', 'N/A') for metric in k_overall_metrics]\n"
                '        logger.info("All overall metrics: average_precision: %s", avg_precision_values)\n'
                "    else:\n"
                '        logger.info("All overall metrics: average_precision: []")'
            ),
        }
