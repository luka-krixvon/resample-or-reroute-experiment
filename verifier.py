"""Imperfect-verifier final selection (used by the verifier-quality ablation).

quality q interpolates between a random pick (q=0) and a perfect verifier (q=1):
    P(final correct) = q * 1[any drawn sample correct] + (1-q) * fraction correct.
"""
import numpy as np


def verifier_expected_correct(drawn_correct, quality: float) -> float:
    if len(drawn_correct) == 0:
        return 0.0
    arr = np.asarray(drawn_correct, dtype=float)
    any_correct = float(arr.max())
    frac_correct = float(arr.mean())
    return quality * any_correct + (1.0 - quality) * frac_correct
