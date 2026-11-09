"""Interactive Windows launcher for the existing investigation command."""
import os
import shlex
import sys
from pathlib import Path

from .cli import main as cli_main


def load_openai_settings(path):
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("=")
        name = name.strip()
        if separator and name in {"OPENAI_API_KEY", "OPENAI_MODEL"}:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if value:
                os.environ.setdefault(name, value)


def ask(label, default=""):
    return input(f"{label} [{default or 'none'}]: ").strip() or default


def launch():
    print("Strategy Finder")
    print("Type a new value or press Enter to keep the default in brackets.\n")
    asset = ask("Asset", "BTC")
    pair = ask("Second asset for rotation")
    preset = ask("Run type (quick, full, offline)", "quick").lower()
    if preset not in {"quick", "full", "offline"}:
        raise ValueError("Run type must be quick, full, or offline.")

    frames = ask("Timeframes", "1h 4h 1d" if preset == "full" else "4h").split()
    duration = ask("Duration", "8h" if preset == "full" else "5m")
    variants = ask("Variants per experiment", "1000" if preset == "full" else "12")
    experiments = ask("Experiment limit (0 runs until deadline)", "0" if preset == "full" else "1")
    matched = ask("Compare one rule change at a time (Y/N)", "N").lower() in {"y", "yes"}

    args = ["investigate", "--asset", asset, "--timeframes", *frames,
            "--duration", duration, "--variants", variants, "--experiments", experiments]
    if pair:
        args.extend(["--pair", pair])
    if matched:
        args.append("--matched")
    if preset == "offline":
        args.append("--offline")
    else:
        load_openai_settings(Path(".env"))
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Add OPENAI_API_KEY to the project .env file, or choose offline.")
        model = ask("OpenAI model", os.environ.get("OPENAI_MODEL") or "gpt-6.1-sol")
        args.extend(["--model", model])

    print("\nstrategy-finder " + shlex.join(args))
    if ask("Start investigation (Y/N)", "Y").lower() not in {"y", "yes"}:
        return 0
    sys.argv[1:] = args
    return cli_main()


if __name__ == "__main__":
    try:
        raise SystemExit(launch())
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        raise SystemExit(130)
    except (OSError, ValueError) as exc:
        print(f"Strategy Finder: {exc}", file=sys.stderr)
        raise SystemExit(1)
