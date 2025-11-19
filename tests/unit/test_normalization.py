import numpy as np
import pytest
import sys
import os



# directory path for --cov
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


from orpl.normalization import minmax, maxband, snv, auc


# fixtures

@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum
    ratios = [0.5, 0.35]
    noiselvls = [0.01, 0.03, 0.05]
    slist = []
    for rf_ratio in ratios:
        for noise in noiselvls:
            s,r,b,n = gen_synthetic_spectrum('nylon', rf_ratio, noise,
                                              baseline_preset='aluminium')
            slist.append((s))
    return slist


@pytest.fixture(scope="module")
def synthetic_nylon_np_array(gen_synthetic_nylon):
    signal = np.asarray(gen_synthetic_nylon)
    return signal



"""
normalization.py normalization module 

"""



# 1. minmax

@pytest.mark.normalization
def test_minmax_fails_with_no_input():
    with pytest.raises(TypeError):
        minmax()


@pytest.mark.normalization
def test_minmax_catch_invalid_input(gen_synthetic_nylon):
    with pytest.raises(AttributeError) as e: # Incorrect call returns attribute error
        minmax("invalid input"), (f"Error was not caugut by the system: {e}")
    # try with ragular list instead of np.ndarray
    with pytest.raises(AttributeError) as e:
        minmax(gen_synthetic_nylon), (f"Error was not caugut by the system: {e}")
    # try multiple correct inputs
    with pytest.raises(TypeError) as e:
        minmax(synthetic_nylon_np_array, synthetic_nylon_np_array), (f"Error was not caugut by the system: {e}")


@pytest.mark.normalization
def test_minmax_correct_type(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    signal_minmax = minmax(signal)
    assert isinstance(signal_minmax, np.ndarray)


@pytest.mark.normalization
def test_minmax_correct_shape(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    signal_minmax = minmax(signal)
    assert signal_minmax.shape == signal.shape


@pytest.mark.normalization
def test_minmax_bounds_accurate(synthetic_nylon_np_array):
    # Test the accuracy of the function
    signal = synthetic_nylon_np_array
    signal_minmax = minmax(signal)
    assert np.isclose(signal_minmax.min(), 0.0, 1e-10)
    assert np.isclose(signal_minmax.max(), 1.0, 1e-10)


@pytest.mark.normalization
def test_minmax_output_different_from_input(synthetic_nylon_np_array):
    # Just in case
    signal = synthetic_nylon_np_array
    signal_minmax = minmax(signal)
    assert not np.array_equal(signal, signal_minmax)


@pytest.mark.normalization
def test_minmax_all_zeros_input():
    signal = np.zeros((5, 100))
    signal_minmax = minmax(signal)
    assert np.all(np.isnan(signal_minmax))  # should return all nan array
    print("\nneeds failsafe?")

@pytest.mark.normalization
def test_minmax_constant_array_input():
    signal = np.full((5, 100), 7.0)
    signal_minmax = minmax(signal)
    assert np.all(np.isnan(signal_minmax))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_minmax_negative_value(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    for i in range(signal.shape[0]):
        signal[np.random.randint(0,signal.shape[0])][np.random.randint(0,signal.shape[1])] = -7 * np.random.default_rng().random()
        signal_minmax = minmax(signal)
        assert np.all(np.isfinite(signal_minmax))  # should return all nan array
        


# 2. maxband

@pytest.mark.normalization
def test_maxband_fails_with_no_input():
    with pytest.raises(TypeError):
        maxband()


@pytest.mark.normalization
def test_maxband_catch_invalid_input(gen_synthetic_nylon, synthetic_nylon_np_array):
    with pytest.raises(AttributeError) as e: # Incorrect call returns attribute error
        maxband("invalid input", 1), (f"Error was not caugut by the system: {e}")
    # try with ragular list instead of np.ndarray
    with pytest.raises(AttributeError) as e:
        maxband(gen_synthetic_nylon, 1), (f"Error was not caugut by the system: {e}")
    # try with invalid array input (2d array)
    signal = np.asarray(gen_synthetic_nylon)
    with pytest.raises(IndexError) as e:
        maxband(signal, 10), (f"Error was not caugut by the system: {e}")
    # typical invalid band_ix
    with pytest.raises(IndexError) as e:
        maxband(signal[0], 10000), (f"Error was not caugut by the system: {e}")
    # try multiple correct inputs
    with pytest.raises(TypeError) as e:
        maxband(synthetic_nylon_np_array, 3, 4), (f"Error was not caugut by the system: {e}")


@pytest.mark.normalization
def test_maxband_correct_type_and_shape(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    for i in range(signal.shape[0]):
        signal_maxband = maxband(signal[i], 10)
        assert signal_maxband.shape == signal[i].shape
        assert isinstance(signal_maxband, np.ndarray)


@pytest.mark.normalization
def test_maxband_bounds_accurate(synthetic_nylon_np_array):
    # Test the accuracy of the function
    signal = synthetic_nylon_np_array
    for i in range(signal.shape[0]):
        band_ix = np.argmax(signal[i])
        signal_maxband = maxband(signal[i], band_ix)
        assert np.isclose(signal_maxband.min(), 0.0, 1e-10)
        assert np.isclose(signal_maxband[band_ix].max(), 1.0, 1e-10)


@pytest.mark.normalization
def test_maxband_output_different_from_input(synthetic_nylon_np_array):
    # Just in case
    signal = synthetic_nylon_np_array
    for i in range(signal.shape[0]):
        band_ix = np.argmax(signal[i])
        signal_maxband = maxband(signal[i], band_ix)
        assert not np.array_equal(signal[i], signal_maxband)


@pytest.mark.normalization
def test_maxband_all_zeros_input():
    signal = np.zeros((100,))
    signal_maxband = maxband(signal, 10)
    assert np.all(np.isnan(signal_maxband))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_maxband_constant_array_input():
    signal = np.full((100,), 7.0)
    signal_maxband = maxband(signal, 10)
    assert np.all(np.isnan(signal_maxband))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_maxband_negative_value(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array 
    for i in range(signal.shape[0]):
        signal[i][np.random.randint(0,signal.shape[1])] = -7 * np.random.default_rng().random()
        signal_minmax = minmax(signal[i])
        assert np.all(np.isfinite(signal_minmax))  # should return all nan array



# 3. snv

@pytest.mark.normalization
def test_snv_fails_with_no_input():
    with pytest.raises(TypeError):
        snv()


@pytest.mark.normalization
def test_snv_catch_invalid_input(gen_synthetic_nylon, synthetic_nylon_np_array):
    with pytest.raises(AttributeError) as e: # Incorrect call returns attribute error
        snv("invalid input"), (f"Error was not caugut by the system: {e}")
    # try with ragular list instead of np.ndarray
    with pytest.raises(AttributeError) as e:
        snv(gen_synthetic_nylon), (f"Error was not caugut by the system: {e}")
    # try multiple correct inputs
    with pytest.raises(TypeError) as e:
        snv(synthetic_nylon_np_array, synthetic_nylon_np_array), (f"Error was not caugut by the system: {e}")


@pytest.mark.normalization
def test_snv_correct_type(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    signal_snv = snv(signal)
    assert isinstance(signal_snv, np.ndarray)


@pytest.mark.normalization
def test_snv_correct_shape(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    signal_snv = snv(signal)
    assert signal_snv.shape == signal.shape


@pytest.mark.normalization
def test_snv_output_different_from_input(synthetic_nylon_np_array):
    # Just in case
    signal = synthetic_nylon_np_array
    signal_snv = snv(signal)
    assert not np.array_equal(signal, signal_snv)


@pytest.mark.normalization
def test_snv_bounds_accurate(synthetic_nylon_np_array):
    # Test the accuracy of the function
    signal = synthetic_nylon_np_array
    signal_snv = snv(signal)
    assert np.isclose(signal_snv.mean(), 0.0, 1e-10)
    assert np.isclose(signal_snv.std(), 1.0, 1e-10)


@pytest.mark.normalization
def test_snv_all_zeros_input():
    signal = np.zeros((5, 100))
    signal_snv = snv(signal)
    assert np.all(np.isnan(signal_snv))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_snv_constant_array_input():
    signal = np.full((5, 100), 7.0)
    signal_snv = snv(signal)
    assert np.all(np.isnan(signal_snv))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_snv_negative_value(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    for i in range(signal.shape[0]):
        signal[np.random.randint(0,signal.shape[0])][np.random.randint(0,signal.shape[1])] = -7 * np.random.default_rng().random()
        signal_snv = snv(signal)
        assert np.all(np.isfinite(signal_snv))  # should return all nan array



# 4. auc

@pytest.mark.normalization
def test_auc_fails_with_no_input():
    with pytest.raises(TypeError):
        auc()


@pytest.mark.normalization
def test_auc_catch_invalid_input(gen_synthetic_nylon, synthetic_nylon_np_array):
    with pytest.raises(AttributeError) as e: # Incorrect call returns attribute error
        auc("invalid input"), (f"Error was not caugut by the system: {e}")
    # try with ragular list instead of np.ndarray
    with pytest.raises(AttributeError) as e:
        auc(gen_synthetic_nylon), (f"Error was not caugut by the system: {e}")
    # try multiple correct inputs
    with pytest.raises(TypeError) as e:
        auc(synthetic_nylon_np_array, synthetic_nylon_np_array), (f"Error was not caugut by the system: {e}")
    

@pytest.mark.normalization
def test_auc_correct_type(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    signal_auc = auc(signal)
    assert isinstance(signal_auc, np.ndarray)


@pytest.mark.normalization
def test_auc_correct_shape(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    signal_auc = auc(signal)
    assert signal_auc.shape == signal.shape


@pytest.mark.normalization
def test_auc_output_different_from_input(synthetic_nylon_np_array):
    # Just in case
    signal = synthetic_nylon_np_array
    signal_auc = auc(signal)
    assert not np.array_equal(signal, signal_auc)


@pytest.mark.normalization
def test_auc_bounds_accurate(synthetic_nylon_np_array):
    # Test the accuracy of the function
    signal = synthetic_nylon_np_array
    signal_auc = auc(signal)
    assert np.isclose(signal_auc.min(), 0.0, 1e-10)
    assert np.isclose(signal_auc.sum(), 1.0, 1e-10)


@pytest.mark.normalization
def test_auc_all_zeros_input():
    signal = np.zeros((5, 100))
    signal_auc = auc(signal)
    assert np.all(np.isnan(signal_auc))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_auc_constant_array_input():
    signal = np.full((5, 100), 7.0)
    signal_auc = auc(signal)
    assert np.all(np.isnan(signal_auc))  # should return all nan array
    print("\nneeds failsafe?")


@pytest.mark.normalization
def test_auc_negative_value(synthetic_nylon_np_array):
    signal = synthetic_nylon_np_array
    for i in range(signal.shape[0]):
        signal[np.random.randint(0,signal.shape[0])][np.random.randint(0,signal.shape[1])] = -7 * np.random.default_rng().random()
        signal_auc = auc(signal)
        assert np.all(np.isfinite(signal_auc))  # should return all nan array





"""
Currently Covered:
- Default Behavior Tests
- Output Accuracy 
- Constant Arrays
- Negative Values

Not Covered:
- Cosmic Rays? (But technically cosmic ray remocal should clear it)
"""