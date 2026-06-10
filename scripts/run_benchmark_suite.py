import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="벤치마크 suite JSON 파일을 실행한다.")
    parser.add_argument("--suite", type=Path, default=Path("configs/suites/whisper_engine_comparison.json"))
    parser.add_argument("--only", nargs="*", help="Run only selected experiment names or engine names.")
    parser.add_argument("--result_root", type=Path)
    parser.add_argument("--manifest_path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpus")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--no_evaluate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def add_optional(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([f"--{name}", str(value)])


def add_flag(command: list[str], name: str, enabled: bool) -> None:
    if enabled:
        command.append(f"--{name}")


def selected(entry: dict, only: list[str] | None) -> bool:
    if not only:
        return True
    return entry.get("name") in only or entry.get("engine") in only


def build_command(entry: dict, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        entry["script"],
        "--config",
        entry["config"],
    ]
    if entry.get("experiment"):
        command.extend(["--experiment", entry["experiment"]])

    add_optional(command, "result_root", args.result_root)
    add_optional(command, "manifest_path", args.manifest_path)
    add_optional(command, "limit", args.limit)
    add_optional(command, "gpus", args.gpus)
    add_flag(command, "no_resume", args.no_resume)
    add_flag(command, "no_evaluate", args.no_evaluate)
    for option, value in entry.get("options", {}).items():
        add_optional(command, option, value)
    return command


def run_suite(args: argparse.Namespace) -> None:
    suite = read_json(args.suite)
    entries = [entry for entry in suite.get("experiments", []) if selected(entry, args.only)]
    if not entries:
        raise ValueError(f"No suite entries selected from {args.suite}")

    failures = []
    for entry in entries:
        label = entry.get("name") or entry.get("engine") or entry["script"]
        command = build_command(entry, args)
        print(f"[suite] {label}: {' '.join(command)}")
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            failures.append({"name": label, "returncode": completed.returncode})
            if not args.continue_on_error:
                break

    if failures:
        raise RuntimeError(f"Suite failed: {failures}")


def main() -> None:
    run_suite(parse_args())


if __name__ == "__main__":
    main()
