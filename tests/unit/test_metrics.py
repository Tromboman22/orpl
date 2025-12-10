import numpy as np
import pytest
import sys
import os


# directory path for --cov
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


from orpl.metrics import raman_snr, assi


# fixtures


@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum

    ratios = [0.5, 0.35]
    noiselvls = [0.01, 0.03, 0.05]
    slist = []
    for rf_ratio in ratios:
        for noise in noiselvls:
            s, r, b, n = gen_synthetic_spectrum(
                "nylon", rf_ratio, noise, baseline_preset="aluminium"
            )
            slist.append((r, b))
    return slist


@pytest.fixture(scope="module")
def synthetic_nylon_zip(gen_synthetic_nylon):
    raman, baseline = zip(*gen_synthetic_nylon)
    raman = np.asarray(raman)
    baseline = np.asarray(baseline)
    return raman, baseline


"""
metrics.py testing module 

"""


