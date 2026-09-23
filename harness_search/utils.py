from enum import Enum
from typing import List


class ProviderFormat(Enum):
    """Supported provider formats"""

    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    QWEN_MOONSHOT = "qwen"
    ANTHROPIC = "anthropic"
    OPENAI_HARMONY = "openai_harmony"


def log_recall_histogram(
    values: List[float],
    label: str,
    logger,
    bins: int = 10,
    width: int = 40,
) -> None:
    """Log a terminal-friendly ASCII histogram of recall values.

    Args:
        values: List of recall values (0.0 to 1.0).
        label: Label for the histogram (e.g., "trajectory_recall", "output_recall").
        logger: Structlog logger instance.
        bins: Number of bins for the histogram.
        width: Maximum width of the histogram bars in characters.
    """
    if not values:
        logger.info(f"{label}_histogram", message="No data to plot")
        return

    # Create histogram bins (0.0 to 1.0)
    bin_edges = [i / bins for i in range(bins + 1)]
    counts = [0] * bins

    for v in values:
        # Clamp to [0, 1] and find bin
        v = max(0.0, min(1.0, v))
        bin_idx = min(int(v * bins), bins - 1)
        counts[bin_idx] += 1

    max_count = max(counts) if counts else 1
    mean_val = sum(values) / len(values)

    # Print histogram directly to terminal for proper formatting
    print()
    print(f"  {label} distribution (n={len(values)}, mean={mean_val:.3f})")
    print("  " + "-" * (width + 20))

    for i, count in enumerate(counts):
        low, high = bin_edges[i], bin_edges[i + 1]
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  [{low:.1f}-{high:.1f}) | {bar:<{width}} | {count}")

    print("  " + "-" * (width + 20))
    print()

    # Log summary stats via structlog
    logger.info(
        f"{label}_histogram",
        n=len(values),
        mean=round(mean_val, 3),
        min=round(min(values), 3),
        max=round(max(values), 3),
    )
