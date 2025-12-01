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

# General tests


@pytest.mark.metrics
def test_metrics_imports_numpy():
    from orpl import metrics

    assert hasattr(metrics, "np")
    assert metrics.np is np


# 1. raman_snr


@pytest.mark.metrics
def test_raman_snr_catch_absence_of_input():
    with pytest.raises(TypeError) as e:  # Incorrect call returns type error
        raman_snr(), (f"Error was not caugut by the system: {e}")


@pytest.mark.metrics
def test_raman_snr_catch_invalid_input(gen_synthetic_nylon):
    with pytest.raises(AttributeError) as e:
        raman_snr("this", "test", "should", "fail"), (
            f"Error was not caugut by the system: {e}"
        )
    # should also fail when inputting normal array
    raman, baseline = zip(*gen_synthetic_nylon)
    # keep as list to raise error
    with pytest.raises(AttributeError) as e:
        raman_snr(raman, baseline, 1, 1), (f"Error was not caugut by the system: {e}")


@pytest.mark.metrics
def test_raman_snr_correct_shape(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    assert raman_snr(raman, baseline, 1.0, 10.0).shape == (6,)


@pytest.mark.metrics
def test_raman_snr_single_array_input_works(gen_synthetic_nylon):
    raman, baseline = zip(*gen_synthetic_nylon)
    raman = np.asarray([raman[0]])  # 2D array with single spectrum
    baseline = np.asarray([baseline[0]])
    assert raman_snr(raman, baseline, 1.0, 1.0)


@pytest.mark.metrics
def test_raman_snr_1D_array_catch_error(gen_synthetic_nylon):
    raman, baseline = zip(*gen_synthetic_nylon)
    raman = np.asarray(raman[0])  # 2D array with single spectrum
    baseline = np.asarray(baseline[0])
    with pytest.raises(np.exceptions.AxisError) as e:
        raman_snr(
            raman, baseline, 1.0, 1.0
        ), f"Numpy Axis error not returned with 1D array input: {e}"


@pytest.mark.metrics
def test_raman_snr_output_is_correct_type(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    # check that the output gives the correct type
    test = raman_snr(raman, baseline, 1.0, 1.0)
    for i in range(6):
        assert type(test[i]) == np.float64


@pytest.mark.metrics
def test_raman_snr_computes_average_proprely():
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


@pytest.mark.metrics
def test_raman_snr_with_partial_zeros(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    # 1. Power or time at 0.0
    settings_zeros = raman_snr(raman, baseline, 0.0, 0.0)
    assert np.all(settings_zeros == 0)
    # 2. Raman alone at 0
    raman = np.zeros_like(raman)
    raman_zeros = raman_snr(raman, baseline, 1.0, 10.0)
    assert np.all(raman_zeros == 0)


@pytest.mark.metrics
def test_raman_snr_with_all_zeros(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    raman = np.zeros_like(raman)
    baseline = np.zeros_like(baseline)
    r_and_b_zeros = raman_snr(raman, baseline, 1.0, 10.0)
    assert np.all(np.isnan(r_and_b_zeros))
    # 3. Everything at 0
    all_zero = raman_snr(raman, baseline, 0.0, 0.0)
    assert np.all(np.isnan(all_zero))
    print("\nRead comments in test file", end="")
    # Should reading ever returns a 0 value by some chance, there is no failsafe code will just return faulty output


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
    print("\nRead comments in test file", end="")
    # --> Should the final test not fail? No failsafe for negative input, error just allowed to freely enter the function


@pytest.mark.metrics
def test_raman_snr_negative_values_spectra():
    raman = np.full((6, 1000), -1)
    baseline = raman
    print("\nRead comments in test file", end="")
    assert np.all(np.isnan(raman_snr(raman, baseline, 1.0, 10.0)))
    # --> Should the test not fail? No failsafe for negative input, error just allowed to freely enter the function


# 2. assi


@pytest.mark.metrics
def test_assi_catches_absence_of_inputs():
    with pytest.raises(TypeError) as e:
        assi(), (f"Error not caught by system: {e}")


@pytest.mark.metrics
def test_assi_catch_invalid_input():
    with pytest.raises(AttributeError) as e:
        assi("invalid")


@pytest.mark.metrics
def test_assi_returns_numpy_float(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    assert type(raman) == np.ndarray
    assert type(assi(raman)) == np.float64


@pytest.mark.metrics
def test_assi_output_is_within_bounds(synthetic_nylon_zip):
    raman, baseline = synthetic_nylon_zip
    test_val = assi(raman)
    # check bounds
    assert test_val > -1.0, f"Assi returns float smaller than -1"
    assert test_val < 1.0, f"Assi returns float larger than 1"


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


@pytest.mark.metrics
def test_assi_with_constant_array():
    raman = np.full((6, 1000), 0.5)
    print("\nRead comments in test file", end="")
    assert np.isnan(assi(raman))
    # Once again division by zero, function runs smoothly and breaks inside the normalization.snv() function
    # More specifically, when you do func - func.mean() and then func/func.std() this encurrs a division by 0 no matter the input


@pytest.mark.metrics
def test_assi_with_negative_constant_array():
    raman = np.full((6, 1000), -0.5)
    print("\nRead comments in test file", end="")
    assert np.isnan(assi(raman))
    # No flags for negative array here, code returns nan array when constant array is entered because of snv func
    # More specifically, when you do func - func.mean() and then func/func.std() this encurrs a division by 0 no matter the input


@pytest.mark.metrics
def test_assi_with_negative_constant_array():
    raman = np.full((6, 1000), 0)
    print("\nRead comments in test file", end="")
    assert np.isnan(assi(raman))
    # No flags for empty array here, code returns nan array when constant array is entered because of snv func
    # More specifically, when you do func - func.mean() and then func/func.std() this encurrs a division by 0 no matter the input
