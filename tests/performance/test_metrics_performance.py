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


# 1. raman_snr


@pytest.mark.metrics
def test_raman_snr_computes_average_properly():
    # Small, easy test array
    raman = np.array([[10, 12, 11], [8, 9, 10]])
    baseline = np.array([[2, 2, 2], [1, 1, 1]])
    test = raman_snr(raman, baseline, 1.0, 1.0)
    # averages computed properly
    expected_raman_avg = np.array([11.0, 9.0])
    expected_baseline_avg = np.array([2.0, 1.0])
    nb_spectrum = 2
    # Using the equation from within the function
    expected_snr = (
        np.sqrt(nb_spectrum * 1.0 * 1.0)
        * expected_raman_avg
        / np.sqrt(expected_raman_avg + expected_baseline_avg)
    )
    assert np.allclose(test, expected_snr, rtol=1e-10)


@pytest.mark.metrics
def test_raman_snr_handles_different_ratios(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    test_low = raman_snr(raman, baseline, 0.5, 10.0)
    test_high = raman_snr(raman, baseline, 1.5, 50.0)
    expected_ratio = np.sqrt((0.5 * 10) / (1.5 * 50))
    assert np.allclose(test_low.mean() / test_high.mean(), expected_ratio, rtol=1e-5)


# 2. assi


@pytest.mark.metrics
def test_assi_output_is_accurate(synthetic_nylon_zip):
    # check that value is correct
    from orpl.normalization import snv

    raman, baseline = synthetic_nylon_zip
    test_val = assi(raman)
    raman_ = snv(raman)
    deviation_sign = np.sign(raman_)
    deviation2 = (raman_) ** 2
    quality_factor = (deviation_sign * deviation2).mean()
    assert np.allclose(test_val, quality_factor, rtol=1e-10)