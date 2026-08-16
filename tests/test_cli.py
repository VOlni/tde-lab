from click.testing import CliRunner

from tde_lab.cli import cli


def test_help_lists_commands():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("compare", "sweep-gaussian", "sweep-sas", "wav", "gui"):
        assert command in result.output


def test_compare_smoke():
    result = CliRunner().invoke(
        cli,
        [
            "compare", "-m", "standard", "--no-save",
            "--frag-length", "256", "--frags", "4",
            "--noise", "gaussian", "--variance", "0.1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Standard FFT" in result.output
    assert "Delay (ms)" in result.output
