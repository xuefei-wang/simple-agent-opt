from prompts.task_prompts import TaskPrompts

class MedSAMSegmentationPromptsWithSkeleton(TaskPrompts):
    """Task prompts for MedSAM segmentation analysis. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown

    This is large-scale medical image segmentation dataset covering the
    dermoscopy/xray modality. The images have dimensions (H, W, C) = (height, width, channel).
    ```
    """

    postprocessing_skeleton_filename = "medsam_segmentation_expert_postprocessing_skeleton.py.txt"

    def get_task_details(self):
        return  f"""
        All of you should work together to write {self.k_word} preprocessing and postprocessing function pairs to improve segmentation performance.
        We provided APIs for both preprocessing and postprocessing functions. You should use functions from useful libraries including but not limited to OpenCV, NumPy, Skimage, Scipy, to implement novel and effective functions.
        1. Based on previous preprocessing and postprocessing functions and their performance (provided below), suggest {self.k_word} new unique function pairs using.
        2. The environment will handle all data loading, evaluation, and logging of the results. Your only job is to write the preprocessing and postprocessing functions.
        3. Do not terminate the conversation until the new functions are evaluated and the numerical performance metrics are logged.
        4. For this task, if all {self.k_word} functions are evaluated correctly, only one iteration is allowed, even if the performance is not satisfactory.
        5. Do not terminate the conversation until the new functions are evaluated and the numerical performance metrics are logged.
        6. Extremely important: Do not terminate the conversation until each of the {self.k_word} new function pairs are evaluated AND their results are written to the function bank.
        7. Recall, this is a STATELESS kernel, so all functions, imports, etc. must be provided in the script to be executed. Any history between previous iterations exists solely as provided preprocessing functions and their performance metrics.
        8. Do not write any code outside of the preprocessing and postprocessing functions.
        9. For preprocessing, the images after preprocessing must still conform to the format specified in the ImageData API. Maintenance of channel identity is critical and channels should not be merged. For postprocessing, it is also critical to maintain the output format as the sample function provided.
        """

    def get_pipeline_metrics_info(self):
        return f"""
    The following metrics are used to evaluate the performance of the pipeline: dsc_metric, nsd_metric.
    - The `dsc_metric` is the dice similarity coefficient (DSC) score of the pipeline and is similar to IoU, measuring the overlap between predicted and ground truth masks.
    - The `nsd_metric` is the normalized surface distance (NSD) score and is more sensitive to distance and boundary calculations.
    """

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, checkpoint_path, k, k_word):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            function_bank_path=function_bank_path,
            checkpoint_path=checkpoint_path,
            k=k,
            k_word=k_word,
        )

    def get_template_replacements(self) -> dict:
        return {
            "task_extra_imports": "import cv2 as cv",
            "task_imports": "    from src.medsam_segmentation import MedSAMTool",
            "task_extra_config": 'checkpoint_path = r"{checkpoint_path}"',
            "task_setup": (
                '    logger.info("Initializing MedSAMTool...")\n'
                '    segmenter = MedSAMTool(gpu_id={gpu_id}, checkpoint_path=checkpoint_path)\n'
                '    logger.info(f"Loading data from {data_path}...")\n'
                '    raw_images, boxes, masks = segmenter.loadData(data_path)\n'
                '    logger.info(f"Data loaded: {len(raw_images)} images.")\n'
                '    logger.info(f"[Single Image] Loading single image as a tester...")\n'
                '\n'
                '    # --- Prepare ImageData ---\n'
                '    batch_size = 8\n'
                '    images = ImageData(raw=raw_images,\n'
                '                batch_size=batch_size,\n'
                '                image_ids=[i for i in range(len(raw_images))],\n'
                '                masks=masks,\n'
                '                predicted_masks=masks)\n'
                '\n'
                '    # Run the pipeline on a single image at first so that if the pipeline fails,\n'
                '    # we fail fast.\n'
                '    # --- [Single Image] Prepare ImageData ---\n'
                '    single_image_data = ImageData(raw=images.raw[:1],\n'
                '                batch_size=1,\n'
                '                image_ids=[0],\n'
                '                masks=images.masks[:1],\n'
                '                predicted_masks=images.masks[:1])'
            ),
            "task_predict_single": "segmenter.predict(preprocessed_images, boxes, used_for_baseline=False)",
            "task_postprocess_check_single": (
                "            if not isinstance(k_final_single_img[i], torch.Tensor):\n"
                '                raise TypeError(f"Postprocessing function {i + 1} returned type {type(k_final_single_img[i])}, expected torch.Tensor.")'
            ),
            "task_evaluate_single": "        k_overall_metrics_single_img.append(segmenter.evaluate(pred, single_image_data.masks))",
            "task_predict_full": "segmenter.predict(preprocessed_images, boxes, used_for_baseline=False)",
            "task_postprocess_check_full": (
                "            if not isinstance(k_final[i], torch.Tensor):\n"
                '                raise TypeError(f"Postprocessing function {i + 1} returned type {type(k_final[i])}, expected ...")'
            ),
            "task_evaluate_full": "        k_overall_metrics.append(segmenter.evaluate(pred, images.masks))",
            "task_metrics_log": '    logger.info("All overall metrics: %s", json.dumps(k_overall_metrics if k_overall_metrics else \'N/A\', indent=2))',
        }
