import numpy as np
import pytest
import soundfile as sf

from tde_lab.preprocessing.vad import (
    VADConfig, detect_voice, fragment_voice_mask,
    frame_mask_to_samples, run_length_smooth, vad_method1, vad_method2,
)

SR = 8000


def _burst_signal(n_silence=SR, n_speech=SR, floor=0.01, amp=1.0):
    """Noise floor, then a loud speech-like burst, then noise floor again."""
    silence1 = floor * np.random.randn(n_silence)
    raw = np.random.randn(n_speech + 4)
    speech = amp * np.convolve(raw, np.ones(5) / 5, mode="valid")
    silence2 = floor * np.random.randn(n_silence)
    return np.concatenate([silence1, speech, silence2])


# ── run-length smoother (exact traces of the MATLAB state machine) ───────────

def test_run_length_promotes_run_of_four_with_retro_patch():
    x = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # states: 1→2→3→4(fire, patch i-1, i-2)→5→6→7→8(erase, patch back)
    expected = np.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    np.testing.assert_array_equal(run_length_smooth(x), expected)


def test_run_length_removes_short_bursts():
    x = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    assert not run_length_smooth(x).any()


def test_run_length_bridges_short_gaps():
    x = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    y = run_length_smooth(x)
    assert y[5]          # single gap inside speech is bridged
    assert y[1:].all()


def test_frame_mask_to_samples_length_and_tail():
    mask = np.array([False, True])
    samples = frame_mask_to_samples(mask, 10, 25)
    assert len(samples) == 25
    assert not samples[:10].any()
    assert samples[10:].all()     # tail padded with the last decision


# ── wavelet detectors ─────────────────────────────────────────────────────────

def test_method1_marks_burst():
    # raw method1 flags are noisy in silence (the MATLAB noise estimate is
    # replaced by single unvoiced frames) — run_length_smooth cleans them up;
    # the strict silence check lives in test_detect_voice_sample_mask
    x = _burst_signal()
    frame = 240
    flags = vad_method1(x, frame)
    n_frames = len(flags)
    third = n_frames // 3
    assert flags[third:2 * third].mean() > 0.8      # burst region voiced
    assert flags[:third].mean() < 0.5               # silence mostly unvoiced
    assert flags[third:2 * third].mean() > 2 * flags[:third].mean()


def test_method2_marks_burst():
    x = _burst_signal(floor=1e-7)   # method2 thresholds are absolute
    flags = vad_method2(x, 240)
    n_frames = len(flags)
    third = n_frames // 3
    assert flags[third:2 * third].mean() > 0.8


def test_detect_voice_sample_mask():
    x = _burst_signal()
    mask = detect_voice(x, SR, VADConfig(method="method1"))
    assert mask.shape == x.shape
    mid = mask[len(x) // 3: 2 * len(x) // 3]
    assert mid.mean() > 0.7
    assert mask[: SR // 2].mean() < 0.2


def test_fragment_voice_mask_selects_burst_fragments():
    x = _burst_signal()
    frag_mask = fragment_voice_mask(x, SR, 1024)
    n = len(frag_mask)
    assert frag_mask[n // 3: 2 * n // 3].mean() > 0.7
    assert frag_mask.any() and not frag_mask.all()


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="Unknown VAD method"):
        detect_voice(np.zeros(SR), SR, VADConfig(method="method3"))


# ── integration with the audio pipeline ──────────────────────────────────────

def test_audio_tde_with_vad(tmp_path):
    from tde_lab.experiments.audio_tde import run as run_audio
    from tde_lab.methods import build_method

    delay = 25
    base = _burst_signal()
    ch1 = base + 0.005 * np.random.randn(len(base))
    ch2 = np.roll(base, delay) + 0.005 * np.random.randn(len(base))
    path = tmp_path / "burst.wav"
    data = np.column_stack([ch1, ch2])
    sf.write(path, data / np.abs(data).max() / 2, SR)

    report = run_audio(
        str(path), [build_method("standard")],
        frag_length=1024, vad=True, subsample_refine=False, save=False,
    )
    assert report.vad_mask is not None
    assert 0 < report.kept_frags < report.pair.n_frags
    assert report.results["Standard FFT"].delay_samples == delay
