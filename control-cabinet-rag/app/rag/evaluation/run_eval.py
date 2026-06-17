import json
from pathlib import Path


def load_golden_set(path: str = "app/rag/evaluation/golden_set.jsonl") -> list[dict]:
    records: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


if __name__ == "__main__":
    dataset = load_golden_set()
    print(f"Loaded {len(dataset)} golden examples")
