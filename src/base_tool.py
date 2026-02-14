"""Base protocol for task tool wrappers.

All tool wrappers used in the agent optimization pipeline should conform to
this interface. The framework calls predict() and evaluate() within the
execution template; __init__ handles model loading and device setup.

Use this as a reference when implementing a new tool wrapper. Your class
does NOT need to inherit from this protocol — it only needs to implement
the same method signatures (structural subtyping).

Example:
    class MyTool:
        def __init__(self, device: int = 0, **kwargs):
            # Load model, set device
            ...

        def predict(self, images: ImageData, **kwargs) -> Any:
            # Run inference, return predictions
            ...

        def evaluate(self, predictions, ground_truth) -> dict:
            # Compare predictions to ground truth
            # Must return dict with string keys and float values
            # e.g. {"f1_score": 0.85}
            ...

Notes on the evaluate() return value:
    - The returned dict is stored as "overall_metrics" in the function bank JSON
    - Your TASK_CONFIGS sampling_function must be able to extract a scalar from it
      e.g. lambda x: x['overall_metrics']['f1_score']
    - All values should be floats (or None for failed evaluations)
"""

from typing import Any, Dict, Protocol, runtime_checkable

from src.data_io import ImageData


@runtime_checkable
class BaseTool(Protocol):
    """Protocol defining the interface for task tool wrappers.

    Tool wrappers encapsulate a scientific analysis tool (e.g., Cellpose,
    MedSAM, DeepCell) and expose a uniform predict/evaluate interface for
    the agent optimization loop.
    """

    def predict(self, images: ImageData, **kwargs) -> Any:
        """Run model inference on preprocessed images.

        Args:
            images: Batched image data in H,W,C format.
            **kwargs: Tool-specific arguments (e.g., batch_size, boxes).

        Returns:
            Model predictions in a tool-specific format (masks, coordinates, etc.).
        """
        ...

    def evaluate(self, predictions: Any, ground_truth: Any) -> Dict[str, float]:
        """Evaluate predictions against ground truth.

        Args:
            predictions: Output from predict().
            ground_truth: Ground truth annotations.

        Returns:
            Dict mapping metric names to float values.
            This dict is stored as "overall_metrics" in the function bank.
            The sampling_function in TASK_CONFIGS must be able to extract
            a scalar score from it.
        """
        ...
