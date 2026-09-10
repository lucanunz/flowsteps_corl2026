import math
from statistics import NormalDist

# *Don't* compute_gaussian_confidence_interval(success_rates, confidence=0.95) 
# *Do* wilson_score_interval(successes, total_trials, confidence=0.95)

def wilson_score_interval(successes: int, total_trials: int, confidence: float = 0.95):
    if total_trials <= 0:
        return 0.0, 0.0

    n = total_trials
    p_hat = successes / n
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)

    denominator = 1.0 + (z**2) / n
    center = p_hat + (z**2) / (2.0 * n)
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * n**2))

    lower = (center - radius) / denominator
    upper = (center + radius) / denominator

    return max(0.0, lower), min(1.0, upper)