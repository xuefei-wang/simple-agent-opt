import json
from typing import Callable

def should_include_function(entry: dict) -> bool:
    """
    Determine if a function should be included based on AutoML optimization status.

    KEEP functions that are:
    - optimized (automl_optimized=True)
    - not_improved (automl_superseded=False explicitly set)
    - never_optimized (neither key set)

    EXCLUDE functions that are:
    - superseded (automl_superseded=True) - old replaced version
    - failed_optimization (automl_optimized=False) - new but worse version
    """
    # Exclude old superseded versions
    if entry.get('automl_superseded', False):
        return False

    # Exclude failed optimization attempts (key exists and is False)
    if 'automl_optimized' in entry and not entry['automl_optimized']:
        return False

    # Keep everything else (optimized=True, superseded=False, or never optimized)
    return True

def pretty_print_list(lst: list) -> str:
    if not lst:
        return "    (No entries to display in this section)\n"

    result = ""
    for item_idx, item in enumerate(lst):
        result += f"--- Entry {item_idx + 1} ---\n"
        
        # Explicitly print performance metrics
        if "overall_metrics" in item and isinstance(item["overall_metrics"], dict):
            result += "  Performance:\n"
            for metric_name, metric_value in item["overall_metrics"].items():
                result += f"    {metric_name}: {metric_value}\n"
        elif "overall_metrics" in item: # If overall_metrics is present but not a dict
            result += f"  Performance (raw): {item['overall_metrics']}\n"
        else:
            result += "  Performance: (Not available)\n"
            
        # Print the preprocessing function
        result += f"  Preprocessing Function:\n```python\n{item.get('preprocessing_function', '# Preprocessing function code not found')}\n```\n"
        result += f"  Postprocessing Function:\n```python\n{item.get('postprocessing_function', '# Postprocessing function code not found')}\n```\n\n"
    
    return result

def top_n(function_bank_path: str, sorting_function: Callable[[dict], float], n: int = 5, maximize = True) -> list:
    '''Return top N functions from function bank as a list, excluding superseded and failed_optimization functions'''
    with open(function_bank_path, 'r') as file:
        json_array = json.load(file)
        # Filter out None values, superseded functions, and failed optimizations
        sorted_bank = list(filter(lambda x: sorting_function(x) is not None and should_include_function(x), json_array))
        sorted_bank = sorted(sorted_bank, key=lambda x: sorting_function(x), reverse=maximize)
        return sorted_bank[:n]
    
def worst_n(function_bank_path: str, sorting_function: Callable[[dict], float], n: int = 5, maximize = True) -> list:
    '''Return worst N functions from function bank as a list, excluding superseded and failed_optimization functions'''
    with open(function_bank_path, 'r') as file:
        json_array = json.load(file)
        # Filter out None values, superseded functions, and failed optimizations
        sorted_bank = list(filter(lambda x: sorting_function(x) is not None and should_include_function(x), json_array))
        sorted_bank = sorted(sorted_bank, key=lambda x:sorting_function(x), reverse = not maximize)
        return sorted_bank[:n]

def last_n(function_bank_path: str, n: int = 5) -> list:
    '''Return last N functions from function bank as a list, excluding superseded and failed_optimization functions'''
    with open(function_bank_path, 'r') as file:
        json_array = json.load(file)
        # Filter out superseded functions and failed optimizations
        filtered_array = list(filter(should_include_function, json_array))
        return filtered_array[-n:]