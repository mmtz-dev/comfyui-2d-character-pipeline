"""The CLI surface: every stage is reachable, and `--help` tells the truth."""

from __future__ import annotations

import pytest

from anim import __version__, cli
from anim.stages import STAGE_ORDER

EXPECTED = ["extract", "pose", "rig", "clip", "pick", "review", "strip", "render",
            "post", "publish", "export", "base", "concept", "animate", "bakeoff",
            "palette", "run"]


def test_every_stage_in_the_plan_is_registered():
    assert STAGE_ORDER == EXPECTED


def test_help_lists_every_stage(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])
    assert raised.value.code == 0
    out = capsys.readouterr().out
    for stage in EXPECTED:
        assert stage in out, f"{stage} is missing from `anim --help`"


def test_each_stage_has_its_own_help(capsys):
    for stage in EXPECTED:
        with pytest.raises(SystemExit) as raised:
            cli.main([stage, "--help"])
        assert raised.value.code == 0
        out = capsys.readouterr().out
        assert stage in out


def test_no_stage_is_still_a_stub():
    """The scaffold's stubs printed "not implemented" and returned 2."""
    import importlib

    for stage in EXPECTED:
        module = importlib.import_module(f"anim.stages.{stage}")
        source = (module.run.__doc__ or "") + (module.__doc__ or "")
        assert "NOT IMPLEMENTED" not in source, f"{stage} is still a stub"


def test_the_version_flag_reports_the_package_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_a_stage_is_required(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main([])
    assert raised.value.code != 0


def test_extract_requires_a_unit_and_an_action(capsys):
    with pytest.raises(SystemExit):
        cli.main(["extract", "clip.mp4"])
    assert "--action" in capsys.readouterr().err


def test_the_comfyui_url_is_a_global_flag():
    parser = cli.build_parser()
    args = parser.parse_args(["--comfyui-url", "http://example:1234",
                              "render", "knight_strike"])
    assert args.comfyui_url == "http://example:1234"


def test_the_api_port_default_matches_compose(tmp_path):
    """compose.yml publishes one port; the config must default to the same one,
    and the retired review service on :8710 must be gone."""
    from pathlib import Path

    from anim import config

    compose = (Path(__file__).resolve().parent.parent / "compose.yml").read_text()
    assert f'"0.0.0.0:{config.API_PORT}:{config.API_PORT}"' in compose
    assert f'"--port", "{config.API_PORT}"' in compose
    assert "8710:8710" not in compose
    assert "\n  review:" not in compose


def test_sigterm_unwinds_through_cleanup_instead_of_killing(monkeypatch):
    """Cancel is SIGTERM; it must raise SystemExit so `except BaseException`
    cleanup (dropping a planned attempt) runs rather than dying mid-frame."""
    import signal

    from anim import cli

    installed = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.setdefault(sig, handler))
    monkeypatch.setattr(cli, "build_parser", lambda: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        cli.main([])
    handler = installed[signal.SIGTERM]
    with pytest.raises(SystemExit) as exc:
        handler(signal.SIGTERM, None)
    assert exc.value.code == 128 + signal.SIGTERM
