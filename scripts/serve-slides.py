#!/usr/bin/env python3
"""Serve the slides for local preview, telling the browser not to cache them.

``python3 -m http.server`` sends Last-Modified but no Cache-Control. A browser
then treats a file it has fetched as fresh for a while (a share of the file's
age) and reuses it on a normal reload, so a replaced image or font does not
show up. This server adds ``Cache-Control: no-store`` to every response.

``no-cache`` would not be enough: the browser would revalidate with
If-Modified-Since, and a replacement file that keeps an older modification
time (copied with its original timestamp, for example) would still get 304.
"""

import argparse
import functools
import http.server
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class NoStoreHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that tells the browser not to store any response."""

    def end_headers(self) -> None:
        """Add Cache-Control before the blank line that ends the headers."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Serve the slides without browser caching.")
    parser.add_argument(
        "--directory",
        type=Path,
        default=REPO_ROOT / "slides-jp",
        help="directory to serve (default: slides-jp/ of this repository)",
    )
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    parser.add_argument("--bind", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    return parser.parse_args()


def main() -> int:
    """Serve the directory until interrupted.

    Returns:
        The exit status.
    """
    args = parse_args()
    handler = functools.partial(NoStoreHandler, directory=str(args.directory))
    with http.server.ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Serving {args.directory} at http://{args.bind}:{args.port}/ (Ctrl-C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
