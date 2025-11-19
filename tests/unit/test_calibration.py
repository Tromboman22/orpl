import numpy as np
import pytest
import json
import sys
import os



# directory path for --cov
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


from orpl.calibration import nm2icm, icm2nm, truncate, xaxis_from_ref, xaxis_from_peaks, autogenx, compute_irf


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
    # Load sample tylenol data from JSON
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
calibration.py testing module 

"""



# 1. nm2icm amd icm2nm


@pytest.mark.calibration
def test_conversion_funcs_break_with_no_input():
    with pytest.raises(TypeError) as e:
        nm2icm(), f"Erronious input was allowed by nm2icm: {e}"
    with pytest.raises(TypeError) as e:
        icm2nm(), f"Erronious input was allowed by icm2nm: {e}"


@pytest.mark.calibration
def test_conversion_funcs_with_invalid_inputs():
    with pytest.raises(TypeError) as e:
        nm2icm("invalid"), f"Invalid input was allowed by nm2icm: {e}"
    with pytest.raises(TypeError) as e:
        icm2nm("invalid"), f"Invalid input was allowed by icm2nm: {e}"
    # functions convert any input to np.ndarray, so no need to test for list inputs or integer/float inputs


@pytest.mark.calibration
def test_conversion_funcs_output_shape(gen_synthetic_nylon):
    wavelengths = gen_synthetic_nylon
    icm = nm2icm(wavelengths[0], nm0=785.0)
    reconverted = icm2nm(icm, nm0=785.0)
    assert icm.shape == wavelengths[0].shape, "nm2icm output shape mismatch"
    assert reconverted.shape == wavelengths[0].shape, "icm2nm output shape mismatch"


@pytest.mark.calibration
def test_conversion_funcs_output_type(gen_synthetic_nylon):
    wavelengths = gen_synthetic_nylon
    icm = nm2icm(wavelengths[0], nm0=785.0)
    reconverted = icm2nm(icm, nm0=785.0)
    assert icm.dtype == wavelengths[0].dtype, "nm2icm output type mismatch"
    assert reconverted.dtype == wavelengths[0].dtype, "icm2nm output type mismatch"


@pytest.mark.calibration
def test_conversion_funcs_with_null_inputs():
    empty_array = np.array([])
    icm = nm2icm(empty_array, nm0=785.0)
    reconverted = icm2nm(empty_array, nm0=785.0)
    assert icm.size == 0, "nm2icm did not handle empty array correctly"
    assert reconverted.size == 0, "icm2nm did not handle empty array correctly"
    all_zero_array = np.zeros(1000)
    icm = nm2icm(all_zero_array, nm0=785.0)
    reconverted = icm2nm(all_zero_array, nm0=785.0)
    assert icm.shape == all_zero_array.shape, "nm2icm did not handle all-zero array correctly"
    assert reconverted.shape == all_zero_array.shape, "icm2nm did not handle all-zero array correctly"


# 2. truncate

@pytest.mark.calibration
def test_truncate_breaks_with_no_input():
    with pytest.raises(TypeError) as e:
        truncate(), f"Erronious input was allowed by truncate: {e}"


@pytest.mark.calibration
def test_truncate_with_invalid_inputs():
    with pytest.raises(AttributeError) as e:
        truncate("invalid"), f"Invalid input was allowed by truncate: {e}"
    with pytest.raises(ValueError) as e:
        assert truncate(np.array([1,2,3]), 5, None) == np.array([]), "Excessive start length was not handled correctly by truncate"
    with pytest.raises(AttributeError) as e:
        truncate([1,2,3], None, None), f"Non np.ndarray was allowed: {e}"


@pytest.mark.calibration
def test_truncate_output_shape_and_type(gen_synthetic_nylon):
    wavelengths = gen_synthetic_nylon
    signal = wavelengths[0]
    start, stop = 100, 500
    truncated_signal = truncate(signal, start=start, stop=stop)
    assert truncated_signal.shape == (stop - start,), "truncate output shape mismatch"
    assert truncated_signal.dtype == signal.dtype, "truncate output type mismatch"


# 3. find_npeaks


# 4. x_axis_from_ref

@pytest.mark.calibration
def test_xaxis_from_ref_breaks_with_no_input():
    with pytest.raises(TypeError) as e:
        xaxis_from_ref(), f"Erronious input was allowed by xaxis_from_ref: {e}"


@pytest.mark.calibration
def test_xaxis_from_ref_with_invalid_inputs():
    with pytest.raises(TypeError) as e:
        xaxis_from_ref("invalid"), f"Invalid input was allowed by xaxis_from_ref: {e}"
    with pytest.raises(AttributeError) as e:
        xaxis_from_ref([1,2,3,4,5,6,7,8,9,10], refx=[1,2,4,5,7,8,9], refy=[1,3,4,6,7,9,10], npks=7), f"Invalid list input was allowed by xaxis_from_ref: {e}"


@pytest.mark.calibration
def test_xaxis_from_ref_output_shape_and_type(sample_tylenol_ref, sample_data_tylenol, gen_synthetic_tylenol):
    wavelengths = sample_data_tylenol[0]
    tylenol_x, tylenol_y = sample_tylenol_ref
    wav = gen_synthetic_tylenol
    xaxis = xaxis_from_ref(wavelengths, refx=tylenol_x, refy=tylenol_y, npks=7, deg=2)
    x2 = xaxis_from_ref(wav[0], refx=tylenol_x, refy=tylenol_y, npks=7, deg=2)
    assert xaxis.shape == wavelengths.shape and x2.shape == wav[0].shape, "xaxis_from_ref output shape mismatch"
    assert xaxis.dtype == wavelengths.dtype and x2.dtype == wav[0].dtype, "xaxis_from_ref output type mismatch"


@pytest.mark.calibration
def test_xaxis_from_ref_with_wrong_ref(gen_synthetic_nylon, gen_synthetic_tylenol, sample_nylon_ref, sample_tylenol_ref):
    npks = 7
    wavelengths_nylon = gen_synthetic_nylon
    wavelengths_tylenol = gen_synthetic_tylenol
    refx_nylon, refy_nylon = sample_nylon_ref
    refx_tylenol, refy_tylenol = sample_tylenol_ref
    nylon_wrong = xaxis_from_ref(wavelengths_nylon[0], refx=refx_tylenol, refy=refy_tylenol, npks=npks)
    nylon = xaxis_from_ref(wavelengths_nylon[0], refx=refx_nylon, refy=refy_nylon, npks=npks)
    tylenol_wrong = xaxis_from_ref(wavelengths_tylenol[0], refx=refx_nylon, refy=refy_nylon, npks=npks)
    tylenol = xaxis_from_ref(wavelengths_tylenol[0], refx=refx_tylenol, refy=refy_tylenol, npks=npks)
    assert not np.array_equal(nylon_wrong, nylon), "xaxis_from_ref did not alter nylon spectrum with wrong ref"
    assert not np.array_equal(tylenol_wrong, tylenol), "xaxis_from_ref did not alter tylenol spectrum with wrong ref"


@pytest.mark.calibration
def test_xaxis_from_ref_with_null_inputs(sample_nylon_ref):
    refx, refy = sample_nylon_ref
    empty_array = np.array([])
    with pytest.raises(ValueError) as e:
        xaxis_from_ref(empty_array, refx=refx, refy=refy, npks=7)


# @pytest.mark.calibration
# def test_xaxis_from_ref_with_constant_input(gen_synthetic_nylon, sample_nylon_ref):
#     wavelengths = gen_synthetic_nylon
#     refx, refy = sample_nylon_ref
#     print(wavelengths[0].shape, refx.shape, refy.shape)
#     constant_array = np.full((1000,), 500.0)
#     with pytest.raises(ValueError) as e:
#         xaxis_from_ref(wavelengths[0], refx=constant_array, refy=constant_array, npks=1)
#     with pytest.raises(ValueError) as e:
#         xaxis_from_ref(constant_array, refx=refx, refy=refy, npks=0)


"""
Still need performance tests for accuracy
"""


# 5. xaxis_from_peaks


@pytest.mark.calibration
def test_xaxis_from_peaks_breaks_with_no_input():
    with pytest.raises(TypeError) as e:
        xaxis_from_peaks(), f"Erronious input was allowed by xaxis_from_peaks: {e}"


@pytest.mark.calibration
def test_xaxis_from_peaks_with_invalid_inputs(gen_synthetic_tylenol):
    with pytest.raises(TypeError) as e:
        xaxis_from_peaks("invalid"), f"Invalid input was allowed by xaxis_from_peaks: {e}"
    with pytest.raises(AttributeError) as e:
        xaxis_from_peaks([1,2,3,4,5,6,7,8,9,10], peaks=[1,2,4,5,7,8,9]), f"Invalid list input was allowed by xaxis_from_peaks: {e}"
    with pytest.raises(np._core._exceptions.UFuncTypeError) as e:
        signal = gen_synthetic_tylenol
        xaxis_from_peaks(signal[0], peaks="invalid"), f"Invalid peaks input was allowed by xaxis_from_peaks: {e}"
    with pytest.raises(UnboundLocalError) as e:
        signal = gen_synthetic_tylenol
        xaxis_from_peaks(signal[0], peaks=[]), f"Insufficient peaks input was allowed by xaxis_from_peaks: {e}"
    # with pytest.raises(AttributeError) as e:
    #     print("here")
    #     fakearr = np.full((1000,), np.nan)
    #     xaxis_from_peaks(fakearr, peaks=[100,222,300]), f"NaN input was allowed by xaxis_from_peaks: {e}"


@pytest.mark.calibration
def test_xaxis_from_peaks_output_shape_and_type(gen_synthetic_tylenol, pks_positions_tylenol):
    wavelengths = gen_synthetic_tylenol
    signal = wavelengths[0]
    # assume peaks at known positions for test
    peaks = pks_positions_tylenol
    xaxis = xaxis_from_peaks(signal, peaks=peaks, deg=2)
    assert xaxis[0].dtype == signal.dtype, "xaxis_from_peaks output type mismatch"


@pytest.mark.calibration
def test_xaxis_from_peaks_with_null_inputs(pks_positions_tylenol):
    empty_array = np.array([])
    with pytest.raises(ValueError) as e:
        xaxis_from_peaks(empty_array, peaks=pks_positions_tylenol), f"xaxis_from_peaks did not handle empty array correctly: {e}"


"""
need edge cases?
"""
