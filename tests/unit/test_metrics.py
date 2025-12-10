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
def test_raman_snr_catch_absence_of_input():
    with pytest.raises(TypeError) as e:  # Incorrect call returns type error
        raman_snr(), (f"Error was not caugut by the system: {e}")


@pytest.mark.metrics
def test_raman_snr_catch_invalid_input(gen_synthetic_nylon):
    with pytest.raises(AttributeError) as e:
        raman_snr(
            "this", "test", "should", "fail"
        ), f"Error was not caugut by the system: {e}"
    # should also fail when inputting normal array
    raman, baseline = zip(*gen_synthetic_nylon)
    # keep as list to raise error
    with pytest.raises(AttributeError) as e:
        raman_snr(raman, baseline, 1, 1), (f"Error was not caugut by the system: {e}")
    with pytest.raises(AttributeError) as e:
        raman_snr([1], [1], 1, 1), (f"Error was not caugut by the system: {e}")


@pytest.mark.metrics
def test_raman_snr_correct_shape_and_type(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    test = raman_snr(raman, baseline, 1.0, 1.0)
    assert test.shape == (6,)
    assert test.dtype == np.float64


@pytest.mark.metrics
def test_raman_snr_with_infinite(gen_synthetic_nylon):
    raman, baseline = zip(*gen_synthetic_nylon)
    raman = np.full_like(raman, np.inf)
    baseline = np.full_like(baseline, np.inf)
    assert np.all(np.isnan(raman_snr(raman, baseline, 1.0, 10.0)))


@pytest.mark.metrics
def test_raman_snr_negative_values_settings(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    # Exposure time negative
    assert np.all(np.isnan(raman_snr(raman, baseline, -1.0, 10.0)))
    # Laser power negative
    assert np.all(np.isnan(raman_snr(raman, baseline, 1.0, -10.0)))
    # Both laser power and exposure time negative
    assert np.all(np.isfinite(raman_snr(raman, baseline, -1.0, -10.0)))


@pytest.mark.metrics
def test_raman_snr_negative_values_spectra():
    raman = np.full((6, 1000), -1)
    baseline = raman
    assert np.all(np.isnan(raman_snr(raman, baseline, 1.0, 10.0)))


# 2. assi


@pytest.mark.metrics
def test_assi_catches_absence_of_inputs():
    with pytest.raises(TypeError) as e:
        assi(), (f"Error not caught by system: {e}")


@pytest.mark.metrics
def test_assi_catch_invalid_input():
    with pytest.raises(AttributeError) as e:
        assi("invalid"), (f"Error not caught by system: {e}")
    with pytest.raises(AttributeError) as e:
        assi([1, 2, 3]), (f"Error not caught by system: {e}")


@pytest.mark.metrics
def test_assi_output_type(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    assert type(assi(raman)) == np.float64


@pytest.mark.metrics
def test_assi_output_is_within_bounds(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    for i in range(raman.shape[0]):
        test_val = assi(raman[i])
        # check bounds
        assert test_val > -1.0, f"Assi returns float smaller than -1"
        assert test_val < 1.0, f"Assi returns float larger than 1"


@pytest.mark.metrics
def test_assi_with_constant_array():
    raman = np.full((6, 1000), 0.5)
    assert np.isnan(assi(raman))


@pytest.mark.metrics
def test_assi_with_negative_constant_array():
    raman = np.full((6, 1000), -0.5)
    assert np.isnan(assi(raman))


@pytest.mark.metrics
def test_assi_with_negative_constant_array():
    raman = np.full((6, 1000), 0)
    assert np.isnan(assi(raman))
