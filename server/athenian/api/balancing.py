from typing import Callable, Coroutine

endpoint_weights = {}


def weight(value: float) -> Callable[[Callable[..., Coroutine]], Callable[..., Coroutine]]:
    """Assign some resource consumption value to the decorated endpoint handler."""

    def set_weight(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        func.weight = endpoint_weights.get(func.__name__, value)
        return func

    return set_weight


def extract_handler_weight(handler: Callable[..., Coroutine]) -> float:
    """Look up the resource consumption value of the endpoint handler."""
    while (weight := getattr(handler, "weight", None)) is None:
        try:
            handler = handler.__wrapped__
        except AttributeError:
            weight = 0.1
            break
    return weight
