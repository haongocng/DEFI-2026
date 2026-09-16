"""Convenience entry point; equivalent to the installed ``drap`` command."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from drap.cli import main


if __name__ == "__main__":
    main()
