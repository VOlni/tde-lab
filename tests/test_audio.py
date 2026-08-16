import numpy as np
import pytest
import soundfile as sf

from tde_lab.experiments.audio_tde import run as run_audio
from tde_lab.methods import build_method
from tde_lab.signals.audio import load_wav_pair

SR = 8000
DELAY = 25          # channel 2 lags channel 1 by 25 samples


def _speechlike(n):
    raw = np.random.randn(n + 4)
    return np.convolve(raw, np.ones(5) / 5, mode="valid")


@pytest.fixture
def stereo_wav(tmp_path):
    """Stereo file: ch2 = ch1 delayed by DELAY samples, mild noise on both."""
    n = SR * 4
    base = _speechlike(n)
    ch1 = base + 0.02 * np.random.randn(n)
    ch2 = np.roll(base, DELAY) + 0.02 * np.random.randn(n)
    path = tmp_path / "pair.wav"
    data = np.column_stack([ch1, ch2])
    sf.write(path, data / np.abs(data).max() / 2, SR)
    return str(path)


def test_load_wav_pair_segment_and_axes(stereo_wav):
    pair = load_wav_pair(stereo_wav, frag_length=512, start_s=1.0, duration_s=2.0)
    assert pair.sample_rate == SR
    assert pair.sig1.shape == (512, (2 * SR) // 512)
    assert pair.raw1.shape == (pair.n_frags * 512,)
    # lag axis in seconds from the real rate
    assert pair.lag_axis[512 // 2] == 0.0
    assert pair.dt == pytest.approx(1.0 / SR)


def test_load_wav_pair_channel_selection(stereo_wav, tmp_path):
    # swap channels → delay flips sign downstream; here just check selection works
    pair = load_wav_pair(stereo_wav, frag_length=512, channels=(1, 0))
    ref = load_wav_pair(stereo_wav, frag_length=512, channels=(0, 1))
    np.testing.assert_allclose(pair.sig1, ref.sig2)
    with pytest.raises(ValueError, match="channel"):
        load_wav_pair(stereo_wav, channels=(0, 3))


def test_load_wav_pair_bad_segment(stereo_wav):
    with pytest.raises(ValueError, match="beyond the file end"):
        load_wav_pair(stereo_wav, start_s=100.0)
    with pytest.raises(ValueError, match="too short"):
        load_wav_pair(stereo_wav, frag_length=1024, start_s=0.0, duration_s=0.01)


def test_audio_tde_recovers_interchannel_delay(stereo_wav):
    report = run_audio(
        stereo_wav,
        [build_method("standard")],
        frag_length=1024,
        save=False,
    )
    result = report.results["Standard FFT"]
    assert result.delay_samples == DELAY
    assert result.delay_seconds == pytest.approx(DELAY / SR, rel=1e-6)
    # sub-sample refinement lands within a tenth of a sample
    assert report.refined_delay_samples == pytest.approx(DELAY, abs=0.1)


def test_audio_tde_prefilter_and_distance_method(stereo_wav):
    from tde_lab.config.settings import MethodConfig
    report = run_audio(
        stereo_wav,
        [build_method("standard"), build_method("dist-l1", MethodConfig(lag_limit=50))],
        frag_length=1024,
        prefilter="cwmedian",
        subsample_refine=False,
        save=False,
    )
    assert report.refined_delay_s is None
    for r in report.results.values():
        assert r.delay_samples == DELAY


def test_audio_tde_saves_outputs(stereo_wav, tmp_path):
    report = run_audio(
        stereo_wav,
        [build_method("standard")],
        frag_length=1024,
        duration_s=2.0,
        save=True,
        output_dir=str(tmp_path / "out"),
    )
    assert report.run_dir is not None
    names = {p.name for p in report.run_dir.iterdir()}
    assert {"mcf_comparison.png", "metrics.csv", "summary.txt"} <= names
    assert "sub-sample refined delay" in (report.run_dir / "summary.txt").read_text()


def test_audio_tde_rejects_unknown_prefilter(stereo_wav):
    with pytest.raises(ValueError, match="prefilter"):
        run_audio(stereo_wav, [build_method("standard")], prefilter="wiener", save=False)
