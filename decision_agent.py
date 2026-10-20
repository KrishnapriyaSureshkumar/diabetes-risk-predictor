
def choose_action(probability, threshold=0.5):
    """
    Educational decision policy.
    The threshold is configurable, not a clinical guideline.
    """
    if probability >= threshold:
        return {
            "action": "Further evaluation",
            "reason": (
                "The model output meets or exceeds "
                "the configured decision threshold."
            )
        }

    return {
        "action": "No threshold-triggered alert",
        "reason": (
            "The model output is below the configured "
            "decision threshold. This does not rule out disease."
        )
    }
