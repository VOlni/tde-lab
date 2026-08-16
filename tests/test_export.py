import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tde_lab.config.settings import ExportConfig
from tde_lab.visualization.plots import plot_pabn_vs_gamma, plot_sigma_vs_gamma
from tde_lab.visualization.saver import ResultSaver
from tde_lab.visualization.style import PRESETS, figure_style, line_style


def _dummy_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_save_figure_multi_format(tmp_path):
    export = ExportConfig(formats=("png", "pdf", "svg"), dpi=100)
    saver = ResultSaver(str(tmp_path), "test_run", export)
    paths = saver.save_figure(_dummy_figure(), "my figure/name")

    assert len(paths) == 3
    for path, fmt in zip(paths, ("png", "pdf", "svg")):
        assert path.suffix == f".{fmt}"
        assert path.exists() and path.stat().st_size > 0
        assert "/" not in path.stem  # sanitised


def test_export_config_validation():
    with pytest.raises(ValueError):
        ExportConfig(formats=("png", "tiff")).validate()
    with pytest.raises(ValueError):
        ExportConfig(dpi=0).validate()
    ExportConfig().validate()  # defaults are fine


def test_figure_style_applies_and_restores():
    default_size = matplotlib.rcParams["font.size"]
    with figure_style("paper"):
        assert matplotlib.rcParams["font.size"] == PRESETS["paper"]["font.size"]
        assert matplotlib.rcParams["savefig.dpi"] == 300
    assert matplotlib.rcParams["font.size"] == default_size


def test_figure_style_unknown_preset():
    with pytest.raises(ValueError, match="Unknown style preset"):
        with figure_style("neon"):
            pass


def test_line_style_cycles():
    assert line_style(0) == {"linestyle": "-"}
    assert line_style(1)["marker"] == "+"
    assert line_style(1)["linestyle"] == "--"
    # cycles beyond the list length without error
    assert "linestyle" in line_style(37)


def test_plot_pabn_and_sigma_vs_gamma():
    gammas = [0, 1, 2, 3, 4, 5, 6]
    curves = {
        "Conventional": np.array([0, 5, 20, 45, 70, 85, 95], dtype=float),
        "Euclidean L1": np.array([0, 1, 3, 8, 15, 25, 40], dtype=float),
    }
    fig = plot_pabn_vs_gamma(gammas, curves, alpha=1.6)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_ylabel() == "Pabn"
    assert "1.6" in ax.get_title()
    plt.close(fig)

    fig2 = plot_sigma_vs_gamma(gammas, curves, alpha=2.0)
    assert "sigma" in fig2.axes[0].get_ylabel()
    plt.close(fig2)
