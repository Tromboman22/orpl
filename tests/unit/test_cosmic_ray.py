import numpy as np
import pytest
import sys
import os


# directory path for --cov
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


from orpl.cosmic_ray import crfilter_single, crfilter_multi


# fixtures


@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum

    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s, _, _, _ = gen_synthetic_spectrum(
            preset="nylon", rb_ratio=0.2, noise_std=n_level / 500
        )
        signals.append(s)
    return np.asarray(signals)


@pytest.fixture(scope="module")
def synthetic_nylon_with_cosmic_rays(gen_synthetic_nylon):
    signals = gen_synthetic_nylon.copy()
    # add cosmic rays
    np.random.seed()  # maximize randomness
    signals = signals.copy()
    num_spectra, spectrum_length = signals.shape
    for i in range(num_spectra):
        cr_position = np.random.randint(
            0, spectrum_length
        )  # random position for cosmic ray
        signals[i, cr_position] += np.random.uniform(0.2, 0.6)  # add cosmic ray spike
    return signals


@pytest.fixture(scope="module")
def sample_data_bacon():
    import json

    data = json.load(open("demos/data/samples/bacon/bacon.json"))
    signals = []
    for _ in range(5):
        rand = np.random.randint(0, len(data))
        signals.append(np.asarray(data[rand]["RawSpectra"]).T)
    return signals  # allow to choose random data in tests


@pytest.fixture(scope="module")
def cosmic_rays_bacon(sample_data_bacon):
    signals = sample_data_bacon.copy()
    for s in signals:
        rand_sig = np.random.randint(0, s.shape[0])
        rand_pos = np.random.randint(0, s.shape[1])
        s[rand_sig][rand_pos] += 4e3  # add cosmic ray spike
    return signals


"""
cosmic_ray.py testing module 

"""


# 1. crfilter_single


@pytest.mark.cosmic_ray
def test_crfilter_single_error_on_no_input():
    with pytest.raises(TypeError) as e:  # Incorrect call returns type error
        crfilter_single(), (f"Error was not caught by the system: {e}")
    # with pytest.raises(TypeError) as e: # Missing input returns type error
    #     crfilter_single(np.array([]))


@pytest.mark.cosmic_ray
def test_crfilter_single_invalid_input(synthetic_nylon_with_cosmic_rays):
    with pytest.raises(AttributeError) as e:  # Input a string rather than an array
        crfilter_single("invalud input"), (f"Error was not caugut by the system: {e}")
    with pytest.raises(ValueError) as e:  # Input a 2D array rather than 1D
        fake_arr = np.full((5, 1000), 1.0)
        crfilter_single(fake_arr), (f"Error was not caugut by the system: {e}")
    with pytest.raises(
        AttributeError
    ) as e:  # Input is regular array rather than np.ndarray
        fake_list = [1.0 for i in range(1000)]
        crfilter_single(fake_list), (f"Error was not caugut by the system: {e}")
    with pytest.raises(ValueError) as e:  # Invalid width input
        crfilter_single(synthetic_nylon_with_cosmic_rays, -1), (
            f"Width as negative number: {e}"
        )
    with pytest.raises(ValueError) as e:
        crfilter_single(synthetic_nylon_with_cosmic_rays, 0.5), (f"Width as float: {e}")
    with pytest.raises(ValueError) as e:  # Invalid std_factor input
        crfilter_single(synthetic_nylon_with_cosmic_rays, 3, -0.1), (
            f"std_factor as negative number: {e}"
        )
    with pytest.raises(TypeError) as e:
        crfilter_single(synthetic_nylon_with_cosmic_rays, 3, "invalid"), (
            f"std_factor as string: {e}"
        )
    # with pytest.raises(AttributeError) as e: # Should fail if the array is full of infs or nans
    #     fake_arr = np.asarray([np.inf for i in range(1000)])
    #     crfilter_single(fake_arr), (f"Error was not caugut by the system: {e}")
    #     fake_arr = np.asarray([np.nan for i in range(1000)])
    #     crfilter_single(fake_arr), (f"Error was not caugut by the system: {e}")


@pytest.mark.cosmic_ray  # confirm output same type as input
def test_crfilter_single_correct_type(synthetic_nylon_with_cosmic_rays):
    signal = synthetic_nylon_with_cosmic_rays
    for i in range(signal.shape[0]):
        filtered_signal = crfilter_single(signal[i])
        assert isinstance(
            filtered_signal, np.ndarray
        ), "Output is not of type np.ndarray"


@pytest.mark.cosmic_ray  # confirm output same shape as input
def test_crfilter_single_correct_shape(synthetic_nylon_with_cosmic_rays):
    signal = synthetic_nylon_with_cosmic_rays
    for i in range(signal.shape[0]):
        filtered_signal = crfilter_single(signal[i])
        assert (
            filtered_signal.shape == signal[i].shape
        ), "Output shape does not match input shape"


@pytest.mark.cosmic_ray  # check that cosmic ray removal detects cosmic rays
def test_crfilter_single_detects_cosmic_rays(gen_synthetic_nylon):
    signals = gen_synthetic_nylon.copy()
    # add cosmic rays manually to ensure we know where they are
    for i in range(signals.shape[0]):
        rand = np.random.randint(0, 1000)  # position to add cosmic ray
        signals[i][rand] += np.random.uniform(0.25, 0.5)
        filtered_signal = crfilter_single(signals[i])
        assert filtered_signal[rand] < signals[i][rand], "Cosmic ray was not removed"


@pytest.mark.cosmic_ray
def test_crfilter_single_output_different_from_input(synthetic_nylon_with_cosmic_rays):
    signal = synthetic_nylon_with_cosmic_rays
    for i in range(signal.shape[0]):
        filtered_signal = crfilter_single(signal[i])
        assert not np.array_equal(
            filtered_signal, signal[i]
        ), "Filtered signal is identical to input signal"


@pytest.mark.cosmic_ray
def test_crfilter_single_handles_constant_array():
    constant_sig = np.full(1000, 0.5)
    filtered_signal = crfilter_single(constant_sig)
    assert np.array_equal(
        filtered_signal, constant_sig
    ), "Filtered constant array should be identical to input"
    constant_sig[np.random.randint(0, 1000)] += np.random.uniform(
        0.15, 0.3
    )  # add cosmic ray spike


@pytest.mark.cosmic_ray
def test_crfilter_single_cosmic_ray_edges(gen_synthetic_nylon):
    signals = gen_synthetic_nylon
    for i in range(signals.shape[0]):
        signals[i][0] += 0.5  # at start
        signals[i][-1] += 0.5  # at end
        filtered = crfilter_single(signals[i])
        assert (
            filtered[0] <= signals[i][0] and filtered[-1] <= signals[i][-1]
        ), "Edge cosmic ray not removed properly"


# 2. crfilter_multi


@pytest.mark.cosmic_ray
def test_crfilter_multi_error_on_no_input():
    with pytest.raises(TypeError) as e:  # Incorrect call returns type error
        crfilter_multi(), (f"Error was not caugut by the system: {e}")


@pytest.mark.cosmic_ray
def test_crfilter_multi_invalid_input():
    with pytest.raises(AttributeError) as e:  # Input a string rather than an array
        crfilter_multi("invalud input"), (f"Error was not caugut by the system: {e}")
    with pytest.raises(
        AttributeError
    ) as e:  # Input is regular array rather than np.ndarray
        fake_list = [[1.0 for i in range(1000)] for j in range(5)]
        crfilter_multi(fake_list), (f"Error was not caugut by the system: {e}")


@pytest.mark.cosmic_ray
def test_crfilter_multi_1D_array():
    with pytest.raises(IndexError) as e:  # Input a 1D array rather than 2D
        fake_arr = np.full(1000, 1.0)
        crfilter_multi(fake_arr), (f"Error was not caugut by the system: {e}")


@pytest.mark.cosmic_ray
def test_crfilter_multi_no_cosmic_rays():
    signal = []
    for _ in range(20):
        signal.append([1.0 for i in range(1000)])
    signal = np.asarray(signal)
    copy = crfilter_multi(signal)
    assert np.array_equal(
        copy, signal
    ), "Constant array edge case should get caught by cr_filter_multi"


@pytest.mark.cosmic_ray
def test_crfilter_multi_entire_signal_flagged_cosmic_ray(sample_data_bacon):
    signal = sample_data_bacon[0]
    copy = crfilter_multi(
        signal, width=3, disparity_threshold=-1.0
    )  # very low threshold to flag entire signal
    assert np.array_equal(
        copy, signal
    ), "Entire signal as cosmic ray edge case should return unchanged array"


@pytest.mark.cosmic_ray  # confirm output same type as input
def test_crfilter_multi_correct_type(cosmic_rays_bacon):
    signal = cosmic_rays_bacon[0]
    filtered_signal = crfilter_multi(signal)
    assert isinstance(filtered_signal, np.ndarray), "Output is not of type np.ndarray"


@pytest.mark.cosmic_ray  # confirm output same shape as input
def test_crfilter_multi_correct_shape(cosmic_rays_bacon):
    signal = cosmic_rays_bacon[0]
    filtered_signal = crfilter_multi(signal)
    assert (
        filtered_signal.shape == signal.shape
    ), "Output shape does not match input shape"


@pytest.mark.cosmic_ray
def test_crfilter_multi_output_different_from_input(cosmic_rays_bacon):
    signals = cosmic_rays_bacon
    for i in range(5):  # try 5 different samples
        filtered_signal = crfilter_multi(signals[i])
        assert not np.array_equal(
            filtered_signal, signals
        ), "Filtered signal is identical to input signal"


@pytest.mark.cosmic_ray
def test_crfilter_multi_idempocy(sample_data_bacon):
    # Filter twice to check if same as filtering once
    signals = sample_data_bacon
    for i in range(5):  # try 5 different samples
        filtered_once = crfilter_multi(signals[i])
        filtered_twice = crfilter_multi(filtered_once)
        assert np.array_equal(filtered_once, filtered_twice), "Filter is not idempotent"
