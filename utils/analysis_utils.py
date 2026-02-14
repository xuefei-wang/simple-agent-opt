"""Shared utility functions for analysis scripts (figs/*_analyze_trajectories.py)."""

from typing import List, Dict, Callable
from utils.function_bank_utils import should_include_function


def find_all_metrics(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> List[float]:
    '''Returns a list of metric values from a list of JSON objects.'''
    return [metric_lambda(obj) for obj in json_array]


def find_lowest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> Dict:
    '''Returns object with the lowest metric value from a list of JSON objects.'''
    return min(json_array, key=metric_lambda)


def find_highest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> Dict:
    '''Returns object with the highest metric value from a list of JSON objects.'''
    return max(json_array, key=metric_lambda)


def find_rolling_lowest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> List[float]:
    '''Returns a list of metric values, each index being the lowest value up until that point'''
    rolling_lowest = []
    current_lowest = float('inf')
    for obj in json_array:
        metric_value = metric_lambda(obj)
        if metric_value < current_lowest:
            current_lowest = metric_value
        rolling_lowest.append(current_lowest)
    return rolling_lowest


def find_rolling_highest(json_array: List[Dict], metric_lambda: Callable[[Dict], float]) -> List[float]:
    '''Returns a list of metric values, each index being the highest value up until that point'''
    rolling_highest = []
    current_highest = float('-inf')
    for obj in json_array:
        metric_value = metric_lambda(obj)
        if metric_value > current_highest:
            current_highest = metric_value
        rolling_highest.append(current_highest)
    return rolling_highest


def find_top_k(json_array: List[Dict], metric_lambda: Callable[[Dict], float], k: int = 5) -> List[Dict]:
    '''Returns the top k entries by metric value, excluding superseded and failed_optimization.'''
    filtered_array = [obj for obj in json_array if should_include_function(obj)]
    return sorted(filtered_array, key=metric_lambda, reverse=True)[:k]


def convert_string_to_function(func_str, func_name):
    '''Converts a function string to a callable function object via exec.'''
    namespace = {}
    exec(func_str, {}, namespace)
    return namespace[func_name]


def extract_metric(entry: Dict, metric_name: str, default=None):
    '''Extract a metric value from a function bank entry, handling schema variations.

    Tries these paths in order:
      entry[metric_name][metric_name]
      entry['overall_metrics'][metric_name]
      entry['metrics'][metric_name]
      entry[metric_name]  (direct scalar)
    '''
    paths = [
        lambda e: e[metric_name][metric_name],
        lambda e: e['overall_metrics'][metric_name],
        lambda e: e['metrics'][metric_name],
        lambda e: e[metric_name],
    ]
    for path in paths:
        try:
            val = path(entry)
            # Guard against getting a dict when we want a scalar
            if not isinstance(val, dict):
                return val
        except (KeyError, TypeError):
            continue
    return default


def parse_automl_status(entry: Dict) -> str:
    '''Classify the automl optimization status of a function bank entry.'''
    if entry.get('automl_superseded', False):
        return "superseded"          # EXCLUDE - old replaced version
    elif entry.get('automl_optimized', False):
        return "optimized"           # KEEP - new improved version
    elif 'automl_optimized' in entry:  # Key exists but is False
        return "failed_optimization" # EXCLUDE - new but worse
    elif 'automl_superseded' in entry: # Key exists but is False
        return "not_improved"        # KEEP - original kept because optimization didn't help
    else:
        return "never_optimized"     # KEEP - no optimization attempted
