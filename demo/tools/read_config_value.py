import sys
from pathlib import Path


def parse_simple_yaml(config_path: Path) -> dict:
    data = {}
    stack = [(-1, data)]
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line or line.lstrip().startswith("- "):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_part, value_part = line.strip().split(":", 1)
        key_part = key_part.strip()
        value_part = value_part.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value_part:
            parent[key_part] = {}
            stack.append((indent, parent[key_part]))
            continue
        parent[key_part] = value_part.strip("\"'")
    return data


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return parse_simple_yaml(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def config_value(data: dict, dotted_key: str, default: str) -> object:
    value = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: read_config_value.py <config_path> <dotted_key> <default>")
    config_path = Path(sys.argv[1])
    dotted_key = sys.argv[2]
    default = sys.argv[3]
    print(config_value(load_config(config_path), dotted_key, default))


if __name__ == "__main__":
    main()
