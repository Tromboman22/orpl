import numpy as np
import pytest
import sys
import os


# directory path for --cov
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)


from orpl.baseline_removal import imodpoly, morph_br, bubblefill


# fixtures


@pytest.fixture(scope="module")
def gen_synthetic_nylon():  # gen_slist from demo #4
    from orpl.synthetic import gen_synthetic_spectrum

    signals = []
    for n_level in [0, 1, 5, 10, 15, 30]:
        s, r, b, _ = gen_synthetic_spectrum(
            preset="nylon", rb_ratio=0.2, noise_std=n_level / 500
        )
        signals.append((s, r, b))
    signals = np.asarray(signals)
    sig, ram, bas = zip(*signals)
    return (sig, ram, bas)


@pytest.fixture(scope="module")
def sample_data_bacon():
    import json

    data = json.load(open("../demos/data/samples/bacon/bacon.json"))
    signals = []
    for _ in range(5):
        rand = np.random.randint(0, len(data))
        signals.append(np.asarray(data[rand]["RawSpectra"]).T)
    return signals  # allow to choose random data in tests


"""
baseline_removal.py testing module 

"""


# 1. imodpoly


@pytest.mark.baseline_removal
def test_imodpoly_breaks_with_no_input():
    with pytest.raises(TypeError) as e:
        imodpoly(), f"Erronious input was allowed by imodpoly: {e}"


@pytest.mark.baseline_removal
def test_imodpoly_with_invalid_inputs(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    with pytest.raises(IndexError) as e:
        ram, bas = imodpoly("invalid"), f"Imodpoly should break on string input: {e}"
    with pytest.raises(ValueError) as e:
        fake_arr = np.full((5, 1000), 1.0)
        ram, bas = imodpoly(fake_arr), f"Imodpoly should break on 2d input: {e}"
    with pytest.raises(ValueError) as e:
        ram, bas = (
            imodpoly(signal[0], poly_order=-1),
            f"Imodpoly should break on negative poly_order: {e}",
        )
    with pytest.raises(UnboundLocalError) as e:
        ram, bas = (
            imodpoly(signal[0], poly_order=3, max_iter=0),
            f"Imodpoly should break on negative or zero as max_iter: {e}",
        )


@pytest.mark.baseline_removal
def test_imodpoly_output_shape(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = imodpoly(signal[0])
    assert ram.shape == signal[0].shape, "Imodpoly returned raman of incorrect shape"
    assert bas.shape == signal[0].shape, "Imodpoly returned baseline of incorrect shape"


@pytest.mark.baseline_removal
def test_imodpoly_output_type(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = imodpoly(signal[0])
    assert isinstance(ram, np.ndarray), "Imodpoly returned raman of incorrect type"
    assert isinstance(bas, np.ndarray), "Imodpoly returned baseline of incorrect type"


@pytest.mark.baseline_removal
def test_imodpoly_baseline_below_signal(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = imodpoly(signal[0])
    assert (
        bas.mean() < signal[0].mean()
    ), "Imodpoly returned baseline that exceeds signal values"
    assert (
        ram.mean() < signal[0].mean()
    ), "Imodpoly returned raman that exceeds signal values"


@pytest.mark.baseline_removal
def test_imodpoly_allzero_signal():
    signal = np.zeros((5, 1000))
    ram, bas = imodpoly(signal[0])
    assert np.all(ram == 0), "Imodpoly failed to return zero raman for zero signal"
    assert np.all(bas == 0), "Imodpoly failed to return zero baseline for zero signal"


@pytest.mark.baseline_removal
def test_imodpoly_over_constant_signal():
    signal = np.full((5, 1000), 5.0)
    ram, bas = imodpoly(signal[0])
    assert np.allclose(
        bas, 5.0
    ), "Imodpoly failed to return constant baseline for constant signal"
    assert np.allclose(
        ram, 0.0
    ), "Imodpoly failed to return zero raman for constant signal"


# 2. morph_br


@pytest.mark.baseline_removal
def test_morph_br_breaks_with_no_input():
    with pytest.raises(TypeError) as e:
        morph_br(), f"Erronious input was allowed by morph_br: {e}"


@pytest.mark.baseline_removal
def test_morph_br_with_invalid_inputs(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    import numba as nb

    with pytest.raises(nb.core.errors.TypingError) as e:
        ram, bas = morph_br("invalid", 3), f"morph_br should break on string input: {e}"
    with pytest.raises(ValueError) as e:
        fake_arr = np.full((5, 1000), 1.0)
        ram, bas = morph_br(fake_arr, 3), f"morph_br should break on 2d input: {e}"
    with pytest.raises(ValueError) as e:
        ram, bas = (
            morph_br(signal[0], hws=-1),
            f"morph_br should break on negative hws: {e}",
        )


@pytest.mark.baseline_removal
def test_morph_br_output_shape(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = morph_br(signal[0], hws=3)
    assert ram.shape == signal[0].shape, "morph_br returned raman of incorrect shape"
    assert bas.shape == signal[0].shape, "morph_br returned baseline of incorrect shape"


@pytest.mark.baseline_removal
def test_morph_br_output_type(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = morph_br(signal[0], hws=3)
    assert isinstance(ram, np.ndarray), "morph_br returned raman of incorrect type"
    assert isinstance(bas, np.ndarray), "morph_br returned baseline of incorrect type"


@pytest.mark.baseline_removal
def test_morph_br_baseline_below_signal(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = morph_br(signal[0], hws=3)
    assert (
        bas.mean() < signal[0].mean()
    ), "morph_br returned baseline that exceeds signal values"
    assert (
        ram.mean() < signal[0].mean()
    ), "morph_br returned raman that exceeds signal values"


@pytest.mark.baseline_removal
def test_morph_br_allzero_signal():
    signal = np.zeros((5, 1000))
    ram, bas = morph_br(signal[0], hws=3)
    assert np.all(ram == 0), "morph_br failed to return zero raman for zero signal"
    assert np.all(bas == 0), "morph_br failed to return zero baseline for zero signal"


@pytest.mark.baseline_removal
def test_morph_br_over_constant_signal():
    signal = np.full((5, 1000), 5.0)
    ram, bas = morph_br(signal[0], hws=3)
    assert np.allclose(
        bas, 5.0
    ), "morph_br failed to return constant baseline for constant signal"
    assert np.allclose(
        ram, 0.0
    ), "morph_br failed to return zero raman for constant signal"


# 3. bubblefill


@pytest.mark.baseline_removal
def test_bubblefill_breaks_with_no_input():
    with pytest.raises(TypeError) as e:
        bubblefill(), f"Erronious input was allowed by bubblefill: {e}"


@pytest.mark.baseline_removal
def test_bubblefill_with_invalid_inputs(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    with pytest.raises(np._core._exceptions._UFuncNoLoopError) as e:
        ram, bas = (
            bubblefill("invalid"),
            f"bubblefill should break on string input: {e}",
        )
    with pytest.raises(ValueError) as e:
        fake_arr = np.full((5, 1000), 1.0)
        ram, bas = bubblefill(fake_arr), f"bubblefill should break on 2d input: {e}"
    ram, bas = bubblefill(signal[0], min_bubble_widths=1)
    np.array_equal(ram, signal[0])
    assert np.allclose(
        ram, 0, atol=0.01
    ), "bubblefill failed to return raman equal to zero for negative min_bubble_width"
    with pytest.raises(TypeError) as e:
        ram, bas = (
            bubblefill(signal[0], min_bubble_widths=0.5),
            f"bubblefill should break on float min_bubble_width: {e}",
        )


@pytest.mark.baseline_removal
def test_bubblefill_output_shape(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = bubblefill(signal[0], min_bubble_widths=3)
    assert ram.shape == signal[0].shape, "bubblefill returned raman of incorrect shape"
    assert (
        bas.shape == signal[0].shape
    ), "bubblefill returned baseline of incorrect shape"


@pytest.mark.baseline_removal
def test_bubblefill_output_type(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = bubblefill(signal[0], min_bubble_widths=3)
    assert isinstance(ram, np.ndarray), "bubblefill returned raman of incorrect type"
    assert isinstance(bas, np.ndarray), "bubblefill returned baseline of incorrect type"


@pytest.mark.baseline_removal
def test_bubblefill_baseline_below_signal(gen_synthetic_nylon):
    signal, raman, baseline = gen_synthetic_nylon
    ram, bas = bubblefill(signal[0], min_bubble_widths=3)
    assert (
        bas.mean() < signal[0].mean()
    ), "bubblefill returned baseline that exceeds signal values"
    assert (
        ram.mean() < signal[0].mean()
    ), "bubblefill returned raman that exceeds signal values"


@pytest.mark.baseline_removal
def test_bubblefill_allzero_signal():
    signal = np.zeros((5, 1000))
    ram, bas = bubblefill(signal[0], min_bubble_widths=3)
    assert np.all(ram == 0), "bubblefill failed to return zero raman for zero signal"
    assert np.all(bas == 0), "bubblefill failed to return zero baseline for zero signal"


@pytest.mark.baseline_removal
def test_bubblefill_over_constant_signal():
    signal = np.full((5, 1000), 5.0)
    ram, bas = bubblefill(signal[0], min_bubble_widths=3)
    assert np.allclose(
        bas, 5.0
    ), "bubblefill failed to return constant baseline for constant signal"
    assert np.allclose(
        ram, 0.0
    ), "bubblefill failed to return zero raman for constant signal"
