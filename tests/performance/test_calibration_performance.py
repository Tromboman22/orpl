import numpy as np
import pytest
import json
import sys
import os



# directory path for --cov
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


from orpl.calibration import nm2icm, icm2nm, truncate, find_npeaks, xaxis_from_ref, xaxis_from_peaks, autogenx, compute_irf


# fixtures

@pytest.fixture(scope="module")
def gen_synthetic_tylenol():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum
    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s,_,_,_ = gen_synthetic_spectrum(preset='tylenol',
                                        rb_ratio=0.2,
                                        noise_std=n_level/500)
        signals.append(s)
    signals = np.asarray(signals)
    return signals


@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum
    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s,_,_,_ = gen_synthetic_spectrum(preset='nylon',
                                        rb_ratio=0.2,
                                        noise_std=n_level/500)
        signals.append(s)
    signals = np.asarray(signals)
    return signals


@pytest.fixture(scope="module")
def sample_data_tylenol():
    data = json.load(open('../demos/data/samples/bacon/tylenol.json'))
    signals = []
    for _ in range(5):
        rand = np.random.randint(0, len(data))
        spectra = np.stack(data[rand]['RawSpectra'])
        spectra = spectra.mean(axis=1)  # average multiple spectra
        signals.append(spectra)
    return signals # allow to choose random data in tests


@pytest.fixture(scope="module")
def sample_tylenol_ref():
    # Load reference tylenol data from CSV
    data = np.genfromtxt('../demos/data/samples/tylenol.csv', delimiter=',')
    xaxis = data[1:, 0]
    ref_tylenol_x = nm2icm(xaxis, 785)
    ref_tylenol_r = data[1:, 2]
    return ref_tylenol_x, ref_tylenol_r


@pytest.fixture(scope="module")
def sample_nylon_ref():
     # Load reference nylon data from CSV
    data = np.genfromtxt('../demos/data/samples/nylon.csv', delimiter=',')
    xaxis = data[1:, 0]
    ref_nylon_x = nm2icm(xaxis, 785)
    ref_nylon_r = data[1:, 2]
    return ref_nylon_x, ref_nylon_r


@pytest.fixture(scope="module")
def pks_positions_tylenol():
    pks_pos = [797, 859, 1170, 1238, 1325, 1608, 1644]
    return pks_pos


"""
calibration.py performance module

"""



# 1. nm2icm amd icm2nm

@pytest.mark.calibration
def test_conversion_funcs_correctness(gen_synthetic_nylon):
    wavelengths = gen_synthetic_nylon
    icm = nm2icm(wavelengths[0], nm0=785.0)
    reconverted = icm2nm(icm, nm0=785.0)
    assert np.allclose(wavelengths[0], reconverted), "nm2icm and icm2nm are not consistent"


# 2. truncate

@pytest.mark.calibration
def test_truncate_keep_spectrum_intact(gen_synthetic_nylon):
    wavelengths = gen_synthetic_nylon
    signal = wavelengths[0]
    start, stop = 100, 900
    truncated_signal = truncate(signal, start=start, stop=stop)
    assert np.allclose(signal[start:stop], truncated_signal), "truncate does not keep spectrum intact"


# 3. find_npeaks


# 4. xaxis_from_ref

@pytest.mark.testing
def test_xaxis_from_ref_accuracy(gen_synthetic_tylenol, sample_tylenol_ref, pks_positions_tylenol):
    wavelengths = gen_synthetic_tylenol
    tylenol_x, tylenol_y = sample_tylenol_ref
    for wav in wavelengths:
        xaxis = xaxis_from_ref(wav, refx=tylenol_x, refy=tylenol_y, npks=7, deg=2)
        # check that known peaks are close to reference peaks
        known_peaks_icm = pks_positions_tylenol
        for pk in known_peaks_icm:
            # ensure that the y values at the peak positions are indeed peaks
            for xval in range(len(xaxis)):
                if np.isclose(xaxis[xval], pk):
                    print("found")
                    # compare peak to neighbor values
                    assert wav[xval] > wav[xval-1], f"xaxis_from_ref peak at {pk} cm-1 is not accurate enough"
                    assert wav[xval] > wav[xval+2], f"xaxis_from_ref peak at {pk} cm-1 is not accurate enough"
                    continue



# 5. xaxis_from_peaks


@pytest.mark.testing
def test_xaxis_from_peaks_accuracy(gen_synthetic_tylenol, pks_positions_tylenol):
    wavelengths = gen_synthetic_tylenol
    for wav in wavelengths: # trying all synthetic spectra
        xaxis = xaxis_from_peaks(wav, peaks=np.array(pks_positions_tylenol), deg=2)
        # check that known peaks are close to reference peaks
        known_peaks_icm = pks_positions_tylenol
        for pk in known_peaks_icm:
            # ensure that the y values at the peak positions are indeed peaks
            for xval in range(len(xaxis[0])):
                if np.isclose(xaxis[0][xval], pk):
                    # compare peak to neighbor values
                    print("found")
                    assert wav[xval] > wav[xval-1], f"xaxis_from_ref peak at {pk} cm-1 is not accurate enough"
                    assert wav[xval] > wav[xval+2], f"xaxis_from_ref peak at {pk} cm-1 is not accurate enough"


# test that the xaxis values are all either increasing or decreasing 