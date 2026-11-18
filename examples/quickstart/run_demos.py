"""Play Testuary's offline demonstrations, one after the other, through its
Python API. They need no model, no key and no setup.

    python run_demos.py [OUTPUT_DIR]

Each demonstration writes its records into a new directory below
OUTPUT_DIR. By default each run of the script gets its own new directory
below demos in the current directory, named by its start time.
"""
import sys
import time
from pathlib import Path

from testuary import demo

#: Each demonstration's directory name and what testuary.demo.play takes for it.
DEMONSTRATIONS = (
    ("python", {"language": "python"}),
    ("java", {"language": "java"}),
    ("toy", {"language": "toy"}),
    ("tutorial", {"data": "tutorial"}),
)


def main() -> int:
    root = (Path(sys.argv[1]) if len(sys.argv) > 1
            else Path("demos") / time.strftime("%Y%m%d-%H%M%S"))
    failed = 0
    for name, choice in DEMONSTRATIONS:
        status = demo.play(output_dir=root / name, speed="instant", **choice)
        print(f"{name}: exit status {status}")
        if status == 130:
            return status
        failed = failed or status
    return failed


if __name__ == "__main__":
    sys.exit(main())
