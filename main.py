from dotenv import load_dotenv
import os
import torch
import json
import argparse
import random
from datetime import datetime
import textwrap
import traceback
import subprocess
import platform
import numpy as np

from autogen import Cache, ConversableAgent, GroupChat, GroupChatManager
from autogen.coding import CodeExecutor
from utils.executors import TemplatedLocalCommandLineCodeExecutor

from prompts.task_prompts import TaskPrompts, _PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER
from prompts.agent_prompts import sys_prompt_code_writer
from prompts.automl_prompts import sys_prompt_automl_agent, prepare_automl_prompt, _AUTOML_PARAMETERIZED_FUNCTION_PLACEHOLDER
from prompts.spot_detection_prompts import SpotDetectionPromptsWithSkeleton
from prompts.cellpose_segmentation_prompts import CellposeSegmentationPromptsWithSkeleton
from prompts.medsam_segmentation_prompts import MedSAMSegmentationPromptsWithSkeleton

from utils.function_bank_utils import top_n, last_n, pretty_print_list, worst_n


TASK_CONFIGS = {
    "spot_detection": {
        "prompt_class": SpotDetectionPromptsWithSkeleton,
        "sampling_function": lambda x: x['overall_metrics']['f1_score'],
        "sampling_function_source": "sampling_function = lambda x: x['overall_metrics']['f1_score']",
        "extra_kwargs": {},
    },
    "cellpose_segmentation": {
        "prompt_class": CellposeSegmentationPromptsWithSkeleton,
        "sampling_function": lambda x: x['overall_metrics']['average_precision'],
        "sampling_function_source": "sampling_function = lambda x: x['overall_metrics']['average_precision']",
        "extra_kwargs": {"dataset_size": 100, "batch_size": 16},
    },
    "medSAM_segmentation": {
        "prompt_class": MedSAMSegmentationPromptsWithSkeleton,
        "sampling_function": lambda x: x['overall_metrics']['dsc_metric'] + x['overall_metrics']['nsd_metric'],
        "sampling_function_source": "sampling_function = lambda x: x['overall_metrics']['dsc_metric'] + x['overall_metrics']['nsd_metric']",
        "extra_kwargs": {},
    },
}


# Load environment variables
load_dotenv()


_INT_TO_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def int_to_word(n: int) -> str:
    """Convert small integer to English word for prompt templates."""
    return _INT_TO_WORD.get(n, str(n))


def set_all_seeds(seed: int):
    """Set seeds for all RNGs to improve reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_commit_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def get_environment_info() -> dict:
    """Capture environment info for reproducibility."""
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": get_git_commit_hash(),
    }
    # GPU info
    if torch.cuda.is_available():
        env_info["cuda_version"] = torch.version.cuda
        env_info["gpu_count"] = torch.cuda.device_count()
        env_info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        env_info["cuda_version"] = None
        env_info["gpu_count"] = 0
    # Installed packages
    try:
        result = subprocess.check_output(
            ["pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode().strip()
        env_info["installed_packages"] = result.split("\n")
    except Exception:
        env_info["installed_packages"] = []
    return env_info


def set_up_agents(executor: CodeExecutor, llm_model: str, k, k_word):
    ''' Prepare agents and state transition'''
    code_writer_prompt = sys_prompt_code_writer(k, k_word)
    
    code_executor_agent = ConversableAgent(
        "code_executor_agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={
            "executor": executor
        }, 
        human_input_mode="NEVER",  # Never take human input for this agent
    )
    code_writer_agent = ConversableAgent(
        "code_writer",
        system_message=code_writer_prompt,
        llm_config={
            "config_list": [
                {"model": llm_model, "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
    )
    
    def state_transition(last_speaker, groupchat):
        ''' Transition between speakers in an agent groupchat '''
        messages = groupchat.messages

        if len(messages) <= 1:
            return code_writer_agent

        if last_speaker is code_writer_agent:
            return code_executor_agent
        elif last_speaker is code_executor_agent:
            return code_writer_agent
    
    return code_executor_agent, code_writer_agent, state_transition


def set_up_automl_agents(optuna_executor: CodeExecutor, llm_model: str, n_functions: int):
    ''' Prepare AutoML agents and state transition for hyperparameter optimization'''

    automl_agent = ConversableAgent(
        "automl_agent",
        system_message=sys_prompt_automl_agent(n_functions),
        llm_config={
            "config_list": [
                {"model": llm_model, "api_key": os.environ["OPENAI_API_KEY"]}
            ]
        },
        code_execution_config=False,  # Turn off code execution for this agent.
        human_input_mode="NEVER",
    )

    optuna_executor_agent = ConversableAgent(
        "optuna_executor_agent",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={
            "executor": optuna_executor
        },
        human_input_mode="NEVER",  # Never take human input for this agent
    )

    def automl_state_transition(last_speaker, groupchat):
        ''' Transition between speakers in AutoML optimization '''
        messages = groupchat.messages

        if len(messages) <= 1:
            return automl_agent

        if last_speaker is automl_agent:
            return optuna_executor_agent
        elif last_speaker is optuna_executor_agent:
            # After execution, end the conversation
            return automl_agent

    return automl_agent, optuna_executor_agent, automl_state_transition


# Load openCV function APIs
with open("assets/APIs.txt", "r") as file:
    APIs = file.read()



def prepare_notes_shared(max_rounds):
    notes_shared = f"""
    - Always check the documentation for the available APIs before reinventing the wheel
    - You only have {max_rounds} rounds of each conversation to optimize the preprocessing function.
    - Import all necessary libraries inside the function. If you need to write a helper function, write it inside the main preprocessing or postprocessing function as well.
    - No need to import ImageData, it has already been imported.
    - THE PROVIDED EVALUATION PIPELINE WORKS OUT OF THE BOX, IF THERE IS AN ERROR IT IS WITH THE PREPROCESSING OR POSTPROCESSING FUNCTION.
    """
    return notes_shared



def function_bank_sample(function_bank_path: str, n_top: int, n_worst: int, n_last: int, sorting_function: callable, current_iteration: int, history_threshold: int=0, total_iterations: int=30, maximize = True):
    ''' Returns a sample of the function bank 
    
    Args:
        function_bank_path (str): Path to the function bank
        n_top (int): Number of top performing functions to return
        n_worst (int): Number of worst performing functions to return
        n_last (int): Number of last executed functions to return
        sorting_function (callable): Function to sort the function bank
        current_iteration (int): Current iteration of the pipeline
        history_threshold (int): Threshold for showing the function bank history
        total_iterations (int): Total number of iterations for the pipeline
        maximize (bool): Whether to maximize or minimize the metric
    Returns:
        str: Inlined samples of the function bank
    
    
    Example:
    ```python
    function_bank_sample(function_bank_path, n_top=100, n_worst=0, n_last=0, sorting_function=lambda x: x["F1"])
    ```
    Usage: Return all elements of the function bank, set n_top to a high number
    and n_worst to 0, and n_last to 0.
    
    """
    
    '''
    
    if current_iteration < history_threshold:
        return f"Function bank history will be shown after iteration {history_threshold}, you are currently on iteration {current_iteration} of {total_iterations}"

    sample = ""

    if(n_top > 0):
        sample += f"""
    ## Top {n_top} performing functions from function bank:
    {pretty_print_list(top_n(function_bank_path, n = n_top, sorting_function=sorting_function, maximize=maximize))}

        """
    
    if(n_worst > 0):
        sample += f"""
    ## Worst {n_worst} performing functions from the function bank:
    {pretty_print_list(worst_n(function_bank_path, n = n_worst, sorting_function=sorting_function, maximize=maximize))}

        """

    if(n_last > 0):
        sample += f"""
    ## Execution history / most recent {n_last} functions from function bank:
    {pretty_print_list(last_n(function_bank_path, n = n_last))}

        """

    return sample

def prepare_prompt(notes_shared: str, function_bank_path: str, prompts: TaskPrompts, sampling_function: callable,
                   current_iteration: int, history_threshold: int = 0, total_iterations: int = 30, maximize=True,
                   n_top: int = 5, n_worst: int = 5, n_last: int = 5):

    prompt_pipeline_optimization = textwrap.dedent(f"""\
Your task is to implement {prompts.k_word} pairs of preprocessing and postprocessing functions to optimize the performance of a machine learning pipeline on a specific dataset.
We provided the APIs for both preprocessing and postprocessing functions. You should use functions from useful libraries including but not limited to OpenCV, NumPy, Skimage, Scipy, to implement novel and effective functions.

## Preprocessing Functions API:
```python
# All preprocessing function names should be of the form preprocess_images_i where i enumerates the preprocessing function, beginning at 1
# Import all necessary libraries inside the function, except for ImageData, which has already been imported.
def preprocess_images_i(images: ImageData) -> ImageData:
    import cv2 as cv
    processed_images_list = []
    for img_array in images.raw:
        img_array = np.copy(img_array) # Make a copy of the image array to avoid modifying the original
        processed_img = img_array # Replace with actual processing
        processed_images_list.append(processed_img)
    output_data = ImageData(raw=processed_images_list, batch_size=images.batch_size)
    return output_data
```

## Postprocessing Functions API:
```python
# All postprocessing function names should be of the form postprocess_preds_i where i enumerates the postprocessing function, beginning at 1
# Preprocessing and postprocessing functions should be paired, i.e. preprocess_images_1 with postprocess_preds_1
# Import all necessary libraries inside the function.
{textwrap.dedent(prompts.get_postprocessing_function_api())}
```

## About the dataset: 
{textwrap.dedent(prompts.dataset_info)}

## Task Details:
{textwrap.dedent(prompts.get_task_details())}

## Task Metrics Details:
{textwrap.dedent(prompts.get_pipeline_metrics_info())}

## Function bank sample:
{textwrap.dedent(function_bank_sample(function_bank_path, n_top=n_top, n_worst=n_worst, n_last=n_last, sorting_function=sampling_function, current_iteration=current_iteration, history_threshold=history_threshold, total_iterations=total_iterations))}

## Useful primitive functions API that can be used in the preprocessing and postprocessing functions:
{textwrap.dedent(APIs)}

## Additional Notes:
{textwrap.dedent(notes_shared)}


## Documentation on the `ImageData` class:
```markdown
Framework-agnostic container for batched image data. Handles variable
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
```
""")

    return prompt_pipeline_optimization


def save_chat_history(chat_history, curr_iter, output_folder):
    output_file = os.path.join(output_folder, f"chat_history_ver{curr_iter:03d}.txt")
    with open(output_file, "w") as file:
        for message in chat_history:
            file.write(f"{message['name']}: {message['content']}\n\n")

def save_seed_list(n, file_path, initial_seed):
    """Generates a list of deterministic integer seeds, saves them, and returns the list."""
    # Use Python's random module, seeded once, to generate other seeds
    seed_generator = random.Random(initial_seed)
    # Generate a list of n integers within a reasonable range
    int_seeds = [seed_generator.randint(0, 2**32 - 1) for _ in range(n)]

    with open(file_path, "w") as file:
        for seed_val in int_seeds:
            file.write(f"{seed_val}\n") # Save integer seeds
    print(f"Saved {n} integer seeds based on initial seed {initial_seed} to {file_path}")

    return int_seeds

def save_run_info(args, run_output_dir, num_optim_iter, prompts_instance, cur_time, history_threshold, max_round, llm_model, n_top, n_worst, n_last, k, k_word):
     """Save comprehensive information about the run configuration."""
     # Create a dictionary with all the run information
     run_info = {
         "experiment_name": args.experiment_name,
         "dataset_path": args.dataset,
         "gpu_id": args.gpu_id,
         "random_seed": args.random_seed,
         "timestamp": cur_time,
         "num_optimization_iterations": num_optim_iter,
         "max_rounds": max_round,
         "history_threshold": history_threshold,
         "n_top": n_top,
         "n_worst": n_worst,
         "n_last": n_last,
         "sample_k": k,
         "sample_k_word": k_word,
         "llm_model": llm_model,
         "cache_seed": args.cache_seed,
         "hyperparameter_optimization": args.hyper_optimize,
         "n_hyper_optimize": args.n_hyper_optimize,
         "n_hyper_optimize_trials": args.n_hyper_optimize_trials,
         "hyper_optimize_interval": args.hyper_optimize_interval,
         "cli_args": vars(args),
         "environment": get_environment_info(),
         "prompts_data": {
             "task_specific_prompts": {
                 "dataset_info": prompts_instance.dataset_info,
                 "task_details": prompts_instance.get_task_details(),
                 "pipeline_metrics_info": prompts_instance.get_pipeline_metrics_info(),
             },
             "agent_system_prompts": {
                 "code_writer": sys_prompt_code_writer(k, k_word),
                 "automl": sys_prompt_automl_agent(args.n_hyper_optimize)
             },
             "executable_pipeline_script_template": prompts_instance.run_pipeline_prompt(), # Call the method to get the script string
         }
     }

     # Write the run info to a JSON file
     with open(os.path.join(run_output_dir, "run_info.json"), "w") as file:
         json.dump(run_info, file, indent=4)

def update_run_info_with_end_timestamp(run_output_dir: str):
    """Adds the end timestamp to the run_info.json file."""
    end_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_info_path = os.path.join(run_output_dir, "run_info.json")

    if os.path.exists(run_info_path):
        with open(run_info_path, "r") as file:
            run_info_data = json.load(file)
        
        run_info_data["timestamp_finish"] = end_time
        
        with open(run_info_path, "w") as file:
            json.dump(run_info_data, file, indent=4)
    else:
        print(f"Warning: {run_info_path} not found. Cannot add end timestamp.")



def create_latest_symlink(experiment_output_dir, run_output_dir):
     """Create a symlink to the latest run for easier access."""
     latest_link = os.path.join(experiment_output_dir, "latest")

     # Get just the directory name (timestamp) without the full path
     run_dir_name = os.path.basename(run_output_dir)

     # Remove existing symlink if it exists
     if os.path.islink(latest_link):
         os.unlink(latest_link)

     # Create relative symlink using just the directory name
     try:
         os.symlink(run_dir_name, latest_link)
         print(f"Created symlink: {latest_link} -> {run_dir_name}")
     except Exception as e:
         print(f"Failed to create symlink: {e}")



def run_automl_optimization(iteration_num, seed, function_bank_path, sampling_function,
                            sampling_function_source, kwargs_for_prompt_class,
                            experiment_name, n_hyper_optimize, n_hyper_optimize_trials,
                            work_dir, max_round, llm_model, cache, run_output_dir):
    """Run AutoML hyperparameter optimization on top-performing functions."""
    print(f"Starting AutoML hyperparameter optimization at iteration {iteration_num}")

    try:
        with open(function_bank_path, 'r') as f:
            function_bank = json.load(f)

        if len(function_bank) == 0:
            print("WARNING: Function bank is empty, cannot run AutoML optimization")
            return

        # Only include functions that have never been attempted for optimization
        eligible_functions = [
            (idx, entry) for idx, entry in enumerate(function_bank)
            if 'automl_optimized' not in entry and 'automl_superseded' not in entry
        ]

        # Filter out None values and sort by performance
        eligible_functions = [(idx, entry) for idx, entry in eligible_functions if sampling_function(entry) is not None]
        eligible_functions = sorted(eligible_functions, key=lambda x: sampling_function(x[1]), reverse=True)

        # Get the top N and extract both indices and entries
        top_functions_with_indices = eligible_functions[:n_hyper_optimize]
        source_function_indices = [idx for idx, entry in top_functions_with_indices]
        top_function_entries = [entry for idx, entry in top_functions_with_indices]

        if len(source_function_indices) == 0:
            print("WARNING: No eligible functions for AutoML optimization (all are already optimized or superseded)")
            return

        sampling_func_str = sampling_function_source

        def run_automl_template():
            with open("prompts/automl_execution_template.py.txt", "r") as f:
                template = f.read()

            class TemplateConfig(dict):
                def __missing__(self, key):
                    return "None"

            automl_template_config = {
                "n_trials": n_hyper_optimize_trials,
                "n_fns": len(top_function_entries),
                "source_indices": source_function_indices,
                "experiment_name": experiment_name,
                "seed": seed,
                "sampling_function_code": sampling_func_str,
                "_AUTOML_PARAMETERIZED_FUNCTION_PLACEHOLDER": _AUTOML_PARAMETERIZED_FUNCTION_PLACEHOLDER,
                **kwargs_for_prompt_class,
            }
            formatted_template = template.format_map(TemplateConfig(automl_template_config))
            return formatted_template

        optuna_executor_instance = TemplatedLocalCommandLineCodeExecutor(
            template_script_func=run_automl_template,
            placeholder=_AUTOML_PARAMETERIZED_FUNCTION_PLACEHOLDER,
            work_dir=work_dir,
            timeout=300 * 2.5 * len(top_function_entries) * n_hyper_optimize_trials
        )

        automl_agent, optuna_executor_agent, automl_state_transition = set_up_automl_agents(
            optuna_executor_instance, llm_model, len(top_function_entries)
        )

        automl_group_chat = GroupChat(
            agents=[automl_agent, optuna_executor_agent],
            messages=[],
            max_round=max_round,
            send_introductions=True,
            speaker_selection_method=automl_state_transition,
        )

        automl_group_chat_manager = GroupChatManager(
            groupchat=automl_group_chat,
            llm_config={
                "config_list": [{"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}],
            },
            is_termination_msg=lambda msg: (
                "TERMINATE" in msg["content"] if msg["content"] else False
            ),
        )

        automl_prompt = prepare_automl_prompt(top_function_entries)

        automl_chat_result = automl_agent.initiate_chat(
            automl_group_chat_manager,
            message=automl_prompt,
            summary_method=None,
            cache=cache
        )

        automl_output_file = os.path.join(run_output_dir, f"automl_chat_history_iter_{iteration_num}.txt")
        with open(automl_output_file, "w") as automl_file:
            for message in automl_chat_result.chat_history:
                automl_file.write(f"{message['name']}: {message['content']}\n\n")

    except Exception as e:
        print(f"ERROR: AutoML optimization failed at iteration {iteration_num}: {e}")
        traceback.print_exc()


def main(args: argparse.Namespace):
    # Validate API key early before any setup work
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )

    k = args.k
    k_word = int_to_word(k)
    work_dir = os.getcwd()
    # Get current datetime once at the beginning
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create experiment-specific output directory
    experiment_output_dir = os.path.join(work_dir, args.experiment_name)
    run_output_dir = os.path.join(experiment_output_dir, cur_time)
    # Create directories if they don't exist
    os.makedirs(run_output_dir, exist_ok=True)

    # Update output paths to use the new directory structure
    output_function_bank = os.path.abspath(os.path.join(run_output_dir, "preprocessing_func_bank.json"))
    # Initialize function bank with empty list if it doesn't exist
    if not os.path.exists(output_function_bank):
        with open(output_function_bank, "w") as file:
            json.dump([], file)

    # Configuration from CLI args
    cache_seed = args.cache_seed
    max_round = args.max_round
    llm_model = args.llm_model

    # Set seeds for reproducibility
    set_all_seeds(args.random_seed)
    
    # Load task config
    task_config = TASK_CONFIGS[args.experiment_name]
    prompt_class = task_config["prompt_class"]
    sampling_function = task_config["sampling_function"]
    kwargs_for_prompt_class = {
        "gpu_id": args.gpu_id, "seed": args.random_seed,
        "dataset_path": args.dataset, "function_bank_path": output_function_bank,
        "k": k, "k_word": k_word,
        **task_config["extra_kwargs"],
    }
    if args.checkpoint_path:
        kwargs_for_prompt_class["checkpoint_path"] = args.checkpoint_path
    
    initial_prompts = prompt_class(**kwargs_for_prompt_class)
    save_run_info(args, run_output_dir, args.num_optim_iter, initial_prompts, cur_time, history_threshold=args.history_threshold, max_round=max_round, llm_model=llm_model, n_top=args.n_top, n_worst=args.n_worst, n_last=args.n_last, k=k, k_word=k_word)
    create_latest_symlink(experiment_output_dir, run_output_dir)
    
    seed_list_file = os.path.join(work_dir,"seed_list.txt")
    # Generate seed list
    seed_list = save_seed_list(args.num_optim_iter, seed_list_file, args.random_seed)

    # Run pipeline development and optimization
    with Cache.disk(cache_seed=cache_seed, cache_path_root=f"{work_dir}/cache") as cache:
        
        notes_shared = prepare_notes_shared(max_rounds=max_round)

        for i in range(args.num_optim_iter):
            print(f"\n{'='*60}")
            print(f"Starting iteration {i + 1} of {args.num_optim_iter}")
            print(f"{'='*60}")

            try:
                # Re-seed all RNGs for each iteration for reproducibility
                set_all_seeds(seed_list[i])

                kwargs_for_prompt_class["seed"] = seed_list[i]
                prompts = prompt_class(**kwargs_for_prompt_class)

                executor_instance = TemplatedLocalCommandLineCodeExecutor(
                    template_script_func=prompts.run_pipeline_prompt,
                    placeholder=_PREPROCESSING_POSTPROCESSING_FUNCTION_PLACEHOLDER,
                    work_dir=work_dir,
                    timeout=300 * 2.5 * k
                )

                # Set up agents
                code_executor_agent, code_writer_agent, state_transition = set_up_agents(executor_instance, llm_model, k, k_word)

                group_chat = GroupChat(
                    agents=[
                        code_executor_agent,
                        code_writer_agent,
                    ],
                    messages=[],
                    max_round=max_round,
                    send_introductions=True,
                    speaker_selection_method=state_transition,
                )

                # Initialize group chat manager
                group_chat_manager = GroupChatManager(
                    groupchat=group_chat,
                    is_termination_msg=lambda msg: (
                        "TERMINATE" in msg["content"] if msg["content"] else False
                    ),
                )


                prompt_pipeline_optimization = f"Agent Pipeline Seed {seed_list[i]} \n" + prepare_prompt(notes_shared,
                                                                                                         output_function_bank,
                                                                                                         prompts,
                                                                                                         sampling_function,
                                                                                                         i,
                                                                                                         history_threshold=args.history_threshold,
                                                                                                         total_iterations=args.num_optim_iter,
                                                                                                         n_top=args.n_top,
                                                                                                         n_worst=args.n_worst,
                                                                                                         n_last=args.n_last)

                chat_result = code_executor_agent.initiate_chat(group_chat_manager, message=prompt_pipeline_optimization, summary_method=None,
                                                cache=cache)
                save_chat_history(chat_result.chat_history, i, run_output_dir)

                # Run AutoML optimization at specified intervals
                if args.hyper_optimize and (i + 1) % args.hyper_optimize_interval == 0:
                    run_automl_optimization(
                        iteration_num=i, seed=seed_list[i],
                        function_bank_path=output_function_bank,
                        sampling_function=sampling_function,
                        sampling_function_source=task_config["sampling_function_source"],
                        kwargs_for_prompt_class=kwargs_for_prompt_class,
                        experiment_name=args.experiment_name,
                        n_hyper_optimize=args.n_hyper_optimize,
                        n_hyper_optimize_trials=args.n_hyper_optimize_trials,
                        work_dir=work_dir, max_round=max_round,
                        llm_model=llm_model, cache=cache,
                        run_output_dir=run_output_dir,
                    )

                print(f"Iteration {i + 1} completed successfully.")

            except Exception as e:
                print(f"ERROR: Iteration {i + 1} failed: {e}")
                traceback.print_exc()
                # Save error info for this iteration
                error_file = os.path.join(run_output_dir, f"iteration_{i:03d}_error.txt")
                with open(error_file, "w") as ef:
                    ef.write(f"Iteration {i + 1} failed at {datetime.now().isoformat()}\n")
                    ef.write(f"Seed: {seed_list[i]}\n")
                    ef.write(traceback.format_exc())
                print(f"Continuing to iteration {i + 2}...")
                continue
            finally:
                # Clean up GPU memory between iterations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    update_run_info_with_end_timestamp(run_output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Agent pipeline")
        
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        required=True,
        help="Path to the dataset."
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        choices=["spot_detection", "cellpose_segmentation", "medSAM_segmentation"],
        help="Name of the experiment. Must be one of: spot_detection, cellpose_segmentation, medSAM_segmentation"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the model checkpoint file. Only used for medSAM segmentation."
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use."
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        required=False,
        help="Random seed to use."
    )

    parser.add_argument(
        "--n_top",
        type=int,
        default=3,
        help="Number of top functions to show in the function bank."
    )

    parser.add_argument(
        "--n_worst",
        type=int,
        default=3,
        help="Number of worst functions to show in the function bank."
    )

    parser.add_argument(
        "--n_last",
        type=int,
        default=0,
        help="Number of last functions to show in the function bank."
    )

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError(f"num_optim_iter must be >= 1, got {value}")
        return ivalue

    parser.add_argument(
        "--num_optim_iter",
        type=positive_int,
        default=20,
        help="Number of optimization iterations (must be >= 1, default: 20)."
    )

    parser.add_argument(
        '--hyper_optimize',
        action='store_true',
        help="Whether to run a hyperparameter search after the trial is over."
    )

    parser.add_argument(
        '--n_hyper_optimize',
        type=int,
        default=3,
        help="Number of functions to optimize."
    )

    parser.add_argument(
        '--n_hyper_optimize_trials',
        type=int,
        default=24,
        help="Number of trials for each function to optimize."
    )

    parser.add_argument(
        '--hyper_optimize_interval',
        type=int,
        default=5,
        help="Run hyperparameter optimization every N iterations (default: 5)."
    )

    parser.add_argument(
        "--history_threshold",
        type=int,
        default=0,
        help="The number of iterations to wait before showing the function bank history."
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="gpt-4.1",
        help="LLM model name (default: gpt-4.1). Must be accessible via your OPENAI_API_KEY."
    )

    parser.add_argument(
        "--max_round",
        type=int,
        default=20,
        help="Maximum number of conversation rounds per iteration (default: 20)."
    )

    parser.add_argument(
        "--cache_seed",
        type=int,
        default=4,
        help="Cache seed for AutoGen result caching (default: 4)."
    )

    parser.add_argument(
        "-k",
        type=int,
        default=3,
        choices=range(1, 11),
        metavar="K",
        help="Number of function pairs to generate per iteration (1-10, default: 3)."
    )

    args = parser.parse_args()

    main(args)
