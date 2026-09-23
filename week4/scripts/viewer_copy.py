"""Create the reduced vorticity recording used by the course viewer.

The sampling keeps every second point in both grid directions:

>>> frame = {"t": 1.5, "step": 3, "omega": list(range(16))}
>>> downsample_frame(frame, expected_n=4)
{'t': 1.5, 'step': 3, 'omega': [0, 2, 8, 10]}
>>> downsample_frame({"t": 0, "step": 0, "omega": [1, 2]}, expected_n=4)
Traceback (most recent call last):
...
ValueError: expected 16 omega values, got 2
>>> import io, json
>>> second = {"t": 2.5, "step": 4, "omega": list(range(16, 32))}
>>> source = io.StringIO(json.dumps(frame) + "\\n" + json.dumps(second) + "\\n")
>>> target = io.StringIO()
>>> write_viewer_copy(source, target, expected_n=4)
2
>>> [json.loads(line) for line in target.getvalue().splitlines()]
[{'t': 1.5, 'step': 3, 'omega': [0, 2, 8, 10]}, {'t': 2.5, 'step': 4, 'omega': [16, 18, 24, 26]}]
"""

import json
from pathlib import Path
from typing import TextIO

FULL_GRID_SIZE = 128


def downsample_frame(frame: dict, expected_n: int) -> dict:
    """Return one frame sampled at every second x and y grid point."""
    omega = frame["omega"]
    expected_values = expected_n * expected_n
    if len(omega) != expected_values:
        raise ValueError(f"expected {expected_values} omega values, got {len(omega)}")
    sampled = [
        omega[iy * expected_n + ix]
        for iy in range(0, expected_n, 2)
        for ix in range(0, expected_n, 2)
    ]
    return {"t": frame["t"], "step": frame["step"], "omega": sampled}


def write_viewer_copy(source: TextIO, target: TextIO, expected_n: int) -> int:
    """Stream JSONL frames from source to a reduced JSONL target."""
    count = 0
    for line in source:
        reduced = downsample_frame(json.loads(line), expected_n)
        json.dump(reduced, target, separators=(",", ":"))
        target.write("\n")
        count += 1
    return count


def main() -> None:
    week4 = Path(__file__).resolve().parents[1]
    source_path = week4 / "artifacts" / "random" / "fields.jsonl"
    target_path = week4 / "fields.jsonl"
    with source_path.open(encoding="utf-8") as source, target_path.open(
        "w", encoding="utf-8"
    ) as target:
        frame_count = write_viewer_copy(source, target, FULL_GRID_SIZE)
    print(
        f"wrote {frame_count} frames to {target_path} "
        f"({target_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
