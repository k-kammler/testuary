"""Start Testuary's local web interface through its Python API.

    python start_web.py [DIR] [--no-browser]

It serves DIR, by default the current directory, opens the interface in
the browser unless --no-browser is given, and runs until Ctrl+C. Only
this computer can reach it.
"""
import sys

from testuary import web


def main() -> int:
    arguments = sys.argv[1:]
    open_browser = "--no-browser" not in arguments
    base_dirs = [argument for argument in arguments if argument != "--no-browser"]
    return web.serve(base_dirs[0] if base_dirs else None, open_browser=open_browser)


if __name__ == "__main__":
    sys.exit(main())
