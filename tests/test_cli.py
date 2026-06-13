from zymera.cli import build_parser, normalize_argv
from zymera.pipeline import PHASES


def test_normalize_argv_implies_generate():
    assert normalize_argv(["--phase", "phase1", "--prompt", "x"])[0] == "generate"
    assert normalize_argv(["doctor"]) == ["doctor"]
    assert normalize_argv(["--help"]) == ["--help"]
    assert normalize_argv([]) == []


def test_generate_args_parse():
    args = build_parser().parse_args(
        ["generate", "--phase", "phase2", "--prompt", "p", "--identity", "i",
         "--preset", "fast", "--seed", "7", "--set", "image.steps=5"]
    )
    assert args.phase == "phase2"
    assert args.seed == 7
    assert args.overrides == ["image.steps=5"]


def test_all_phases_accepted_by_parser():
    parser = build_parser()
    for phase in PHASES:
        args = parser.parse_args(["generate", "--phase", phase, "--prompt", "x"])
        assert args.phase == phase


def test_identity_subcommands_parse():
    parser = build_parser()
    args = parser.parse_args(["identity", "create", "abc", "--images", "a.jpg", "b.jpg"])
    assert args.identity_command == "create"
    assert args.images == ["a.jpg", "b.jpg"]
    args = parser.parse_args(["identity", "list"])
    assert args.identity_command == "list"
