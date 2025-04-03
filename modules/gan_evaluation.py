import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
import numpy as np


def ks_test(original, synthetic):
    try:
        ks_stat, p_value = ks_2samp(original, synthetic)
        return {"KS Statistic": ks_stat, "p-value": p_value}
    except Exception as e:
        return {"KS Statistic": None, "p-value": None, "Error": str(e)}

def js_divergence(original, synthetic, bins=30):
    try:
        original_hist, _ = np.histogram(original, bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic, bins=bins, density=True)
        jsd = jensenshannon(original_hist, synthetic_hist)
        return jsd
    except Exception as e:
        return None

def emd(original, synthetic):
    try:
        emd_value = wasserstein_distance(original, synthetic)
        return emd_value
    except Exception as e:
        return None
