import matplotlib.pyplot as plt
import numpy as np
import pytest
import json
import sys
import os


# directory path for --cov
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


from orpl.calibration import (
    nm2icm,
    icm2nm,
    truncate,
    find_npeaks,
    xaxis_from_ref,
    xaxis_from_peaks,
    autogenx,
    compute_irf,
)


# fixtures


@pytest.fixture(scope="module")
def gen_synthetic_tylenol():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum

    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s, _, _, _ = gen_synthetic_spectrum(
            preset="tylenol", rb_ratio=0.2, noise_std=n_level / 500
        )
        signals.append(s)
    signals = np.asarray(signals)
    return signals


@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum

    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s, _, _, _ = gen_synthetic_spectrum(
            preset="nylon", rb_ratio=0.2, noise_std=n_level / 500
        )
        signals.append(s)
    signals = np.asarray(signals)
    return signals


@pytest.fixture(scope="module")
def sample_data_tylenol():
    data = json.load(open("demos/data/samples/bacon/tylenol.json"))
    signals = []
    for _ in range(2):
        rand = np.random.randint(0, len(data) - 1)
        if rand >= 2:  # skip the one with too much noise
            rand += 1
        spectra = np.stack(data[rand]["RawSpectra"])
        spectra = spectra.mean(axis=1)  # average multiple spectra
        signals.append(spectra)
    return signals  # allow to choose random data in tests (out of the 4 options...)


@pytest.fixture(scope="module")
def sample_data_bacon():
    import json

    data = json.load(open("demos/data/samples/bacon/bacon.json"))
    signals = []
    for _ in range(5):
        rand = np.random.randint(0, len(data))
        spectra = np.stack(data[rand]["RawSpectra"])
        spectra = spectra.mean(axis=1)  # average multiple spectra
        signals.append(spectra)
    return signals  # allow to choose random data in tests


@pytest.fixture(scope="module")
def sample_tylenol_ref():
    # Load reference tylenol data from CSV
    data = np.genfromtxt("demos/data/samples/tylenol.csv", delimiter=",")
    xaxis = data[1:, 0]
    ref_tylenol_x = nm2icm(xaxis, 785)
    ref_tylenol_r = data[1:, 2]
    return ref_tylenol_x, ref_tylenol_r


@pytest.fixture(scope="module")
def sample_nylon_ref():
    # Load reference nylon data from CSV
    data = np.genfromtxt("demos/data/samples/nylon.csv", delimiter=",")
    xaxis = data[1:, 0]
    ref_nylon_x = nm2icm(xaxis, 785)
    ref_nylon_r = data[1:, 2]
    return ref_nylon_x, ref_nylon_r


@pytest.fixture(scope="module")
def pks_positions_tylenol():
    pks_pos = [797, 859, 1170, 1238, 1325, 1608, 1644]
    return pks_pos


@pytest.fixture(scope="module")
def pks_positions_nylon():
    pks_pos = [956, 1064, 1132, 1235, 1299, 1443, 1632]
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
    assert np.allclose(
        wavelengths[0], reconverted
    ), "nm2icm and icm2nm are not consistent"


# 2. truncate


@pytest.mark.calibration
def test_truncate_keep_spectrum_intact(gen_synthetic_nylon):
    wavelengths = gen_synthetic_nylon
    signal = wavelengths[0]
    start, stop = 100, 900
    truncated_signal = truncate(signal, start=start, stop=stop)
    assert np.allclose(
        signal[start:stop], truncated_signal
    ), "truncate does not keep spectrum intact"


# 3. find_npeaks


@pytest.mark.calibration
def test_find_npeaks_correctness(gen_synthetic_nylon):
    # default performance test: check that found peaks are actual peaks
    wavelengths = gen_synthetic_nylon[0:3]
    for wav in wavelengths:
        peak_pos = find_npeaks(wav, ntarget=7)
        for pos in peak_pos:
            assert (
                wav[pos] > wav[pos - 1] and wav[pos] > wav[pos + 1]
            ), "find_npeaks did not find actual peaks"


# 4. xaxis_from_ref


@pytest.mark.calibration
def test_xaxis_from_ref_correctness(
    sample_data_tylenol, sample_tylenol_ref, pks_positions_tylenol
):
    # default performance test: check that known peaks are found at expected positions
    wavelengths = sample_data_tylenol
    tylenol_x, tylenol_y = sample_tylenol_ref
    for wav in wavelengths:
        xaxis = xaxis_from_ref(wav, refx=tylenol_x, refy=tylenol_y, npks=7)
        # check that known peaks are close to reference peaks
        known_peaks_icm = pks_positions_tylenol
        exp_peaks = find_npeaks(wav, ntarget=7)
        for val in range(len(known_peaks_icm)):
            # ensure that the peaks are at the expected positions on the x-axis (within 1.5 cm-1)
            assert xaxis[exp_peaks][val] == pytest.approx(
                known_peaks_icm[val], abs=1.5
            ), f"xaxis_from_ref peak at {known_peaks_icm[val]} cm-1 is not accurate enough"
            # ensure that the find_n_peaks found actual peaks
            assert (
                wav[(exp_peaks[val])] > wav[(exp_peaks[val]) - 1]
            ), f"xaxis_from_ref peak at {known_peaks_icm[val]} cm-1 is not accurate enough"
            assert (
                wav[(exp_peaks[val])] > wav[(exp_peaks[val]) + 1]
            ), f"xaxis_from_ref peak at {known_peaks_icm[val]} cm-1 is not accurate enough"


# 5. xaxis_from_peaks


@pytest.mark.calibration
def test_xaxis_from_peaks_accuracy(sample_data_tylenol, pks_positions_tylenol):
    wavelengths = sample_data_tylenol
    print("")
    for wav in wavelengths:  # trying all synthetic spectra
        xaxis, residual = xaxis_from_peaks(
            wav, peaks=np.array(pks_positions_tylenol), deg=2
        )
        # check that known peaks are close to reference peaks
        known_peaks_icm = pks_positions_tylenol
        exp_peaks = find_npeaks(wav, ntarget=7)
        for val in range(len(known_peaks_icm)):
            # ensure that the peaks are at the expected positions on the x-axis (within 1.5 cm-1)
            assert xaxis[exp_peaks][val] == pytest.approx(
                known_peaks_icm[val], abs=1.5
            ), f"xaxis_from_ref peak at {known_peaks_icm[val]} cm-1 is not accurate enough"
            # ensure that the find_n_peaks found actual peaks
            assert (
                wav[(exp_peaks[val])] > wav[(exp_peaks[val]) - 1]
            ), f"xaxis_from_ref peak at {known_peaks_icm[val]} cm-1 is not accurate enough"
            assert (
                wav[(exp_peaks[val])] > wav[(exp_peaks[val]) + 1]
            ), f"xaxis_from_ref peak at {known_peaks_icm[val]} cm-1 is not accurate enough"


@pytest.mark.calibration
def test_xaxis_from_peaks_is_a_function(sample_data_tylenol, pks_positions_tylenol):
    # checks that xaxis_from_peaks produces a valid function when using a controlled input
    wavelengths = sample_data_tylenol
    for wav in wavelengths:
        xaxis, residual = xaxis_from_peaks(
            wav, peaks=np.array(pks_positions_tylenol), deg=2
        )
        values = np.linspace(0, len(xaxis), len(xaxis))
        dx = np.gradient(xaxis, values)
        prod = np.min(dx) * np.max(dx)
        # this checks that the product of max and min of the derivative is positive and non-zero
        assert prod > 0, "xaxis_from_peaks does not produce a valid function "
