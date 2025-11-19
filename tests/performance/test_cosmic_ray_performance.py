import numpy as np
import pytest
import sys
import os



# directory path for --cov
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


from orpl.cosmic_ray import crfilter_single, crfilter_multi


# fixtures

@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum
    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s,_,_,_ = gen_synthetic_spectrum(preset='nylon',
                                        rb_ratio=0.2,
                                        noise_std=n_level/500)
        signals.append(s)
    return np.asarray(signals)


@pytest.fixture(scope="module")
def synthetic_nylon_with_cosmic_rays(gen_synthetic_nylon):
    signals = gen_synthetic_nylon
    # add cosmic rays
    num_spectra, spectrum_length = signals.shape
    for i in range(num_spectra):
        cr_position = np.random.randint(0, spectrum_length)  # random position for cosmic ray
        signals[i, cr_position] += np.random.uniform(0.2, 0.6)  # add cosmic ray spike
    return signals


@pytest.fixture(scope="module")
def sample_data_bacon():
    import json
    data = json.load(open('../demos/data/samples/bacon/bacon.json'))
    signals = []
    for _ in range(5):
        rand = np.random.randint(0, len(data))
        signals.append(np.asarray(data[rand]['RawSpectra']).T)
    return signals # allow to choose random data in tests



"""
cosmic_ray.py performance module 

"""

# 1. crfilter_single

@pytest.mark.cosmic_ray
def test_crfilter_single_is_accurate(gen_synthetic_nylon):
    cr_sig = gen_synthetic_nylon
    for i in range(cr_sig.shape[0]):
        cr_position = np.random.randint(0, len(cr_sig[i]))  # random position for cosmic ray
        cr_sig[i, cr_position] += np.random.uniform(0.2, 0.4)  # add cosmic ray spike
        filtered_signal = crfilter_single(cr_sig[i])
        assert cr_sig[i][cr_position] > filtered_signal[cr_position], "crfilter_single did not accurately remove cosmic rays"

# 2. crfilter_multi

@pytest.mark.cosmic_ray # check that cosmic ray removal detects cosmic rays
def test_crfilter_multi_detects_cosmic_rays(sample_data_bacon):
    data = sample_data_bacon
    # add cosmic rays
    for i in range(5):  # try 10 times to increase difficulty
        cr_signals = data[i]
        rand0 = np.random.randint(0, 20)  # for each spectrum
        rand1 = np.random.randint(0, 1024)  # position to add cosmic ray
        cr_signals[rand0][rand1] += np.random.uniform(1.7e+3, 3.5e+3)
        rand2 = np.random.randint(0, 20)  # for each spectrum
        rand3 = np.random.randint(0, 1024)  # position to add cosmic ray
        cr_signals[rand2][rand3] += np.random.uniform(1.7e+3, 3.5e+3)
        filtered_signals = crfilter_multi(cr_signals, 3, 0.04)
        assert filtered_signals[rand0][rand1] < cr_signals[rand0][rand1], "crfilter_multi did not remove all cosmic rays 5 times in a row"
        assert filtered_signals[rand2][rand3] < cr_signals[rand2][rand3], "crfilter_multi did not remove all cosmic rays 5 times in a row"
