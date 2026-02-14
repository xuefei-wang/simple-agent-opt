from prompts.task_prompts import TaskPrompts


class SpotDetectionPromptsWithSkeleton(TaskPrompts):
    """Task prompts for cell spot detection. Skeletonized version."""

    # --- Define these as CLASS attributes ---
    dataset_info = """
    ```markdown
    This is a single-channel cell spot detection dataset. IMPORTANT: The images have dimensions (B, L, W, C) = (batch, length, width, channel).
    The images have pixel values between 0 and 1 and are in float32 format.
    ```
    """

    postprocessing_skeleton_filename = "spot_detection_expert_postprocessing_skeleton.py.txt"

    def get_task_details(self):
        return f"""
    All of you should work together to write {self.k_word} preprocessing and postprocessing function pairs to improve spot detection performance.
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
    The following metrics are used to evaluate the performance of the pipeline: f1_score.
    f1_score: Mean F1 score of predicted spots
    """

    def __init__(self, gpu_id, seed, dataset_path, function_bank_path, k, k_word):
        super().__init__(
            gpu_id=gpu_id,
            seed=seed,
            dataset_info=self.dataset_info,
            dataset_path=dataset_path,
            function_bank_path=function_bank_path,
            k=k,
            k_word=k_word,
        )

    def get_template_replacements(self) -> dict:
        return {
            "task_extra_imports": "",
            "task_imports": (
                "    from src.spot_detection import DeepcellSpotsDetector\n"
                "    from src.utils import set_gpu_device\n"
                "    import cv2 as cv"
            ),
            "task_extra_config": "set_gpu_device(gpu_id)",
            "task_setup": (
                '    logger.info("Initializing SpotsDetector...")\n'
                '    deepcell_spot_detector = DeepcellSpotsDetector()\n'
                '    logger.info(f"Loading data from {data_path}...")\n'
                '    spots_data = np.load(f"{data_path}", allow_pickle=True)\n'
                '    logger.info(f"Data loaded: {len(spots_data[\'X\'])} images.")\n'
                '    logger.info(f"[Single Image] Loading single image as a tester...")\n'
                '\n'
                '    # --- Prepare ImageData ---\n'
                "    batch_size = spots_data['X'].shape[0]\n"
                "    images = ImageData(raw=spots_data['X'], batch_size=batch_size, image_ids=[i for i in range(batch_size)])\n"
                '\n'
                '    # Run the pipeline on a single image at first so that if the pipeline fails,\n'
                '    # we fail fast.\n'
                '    # --- [Single Image] Prepare ImageData ---\n'
                '    single_image_data = ImageData(images.raw[:2], batch_size=1, image_ids=[0,1])'
            ),
            "task_predict_single": "deepcell_spot_detector.predict(preprocessed_images)",
            "task_postprocess_check_single": "",
            "task_evaluate_single": "        k_overall_metrics_single_img.append(deepcell_spot_detector.evaluate(pred, spots_data['y'][:1]))",
            "task_predict_full": "deepcell_spot_detector.predict(preprocessed_images)",
            "task_postprocess_check_full": "",
            "task_evaluate_full": "        k_overall_metrics.append(deepcell_spot_detector.evaluate(pred, spots_data['y']))",
            "task_metrics_log": '    logger.info("All overall metrics: %s", json.dumps(k_overall_metrics if k_overall_metrics else \'N/A\', indent=2))',
        }
