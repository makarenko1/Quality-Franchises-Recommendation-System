"""
Local-only dev server: serves public/ as static files AND the /api/recommend
route from one process on one port, so the full frontend + API flow can be
exercised locally without vercel dev. Not used by the actual Vercel deployment
(Vercel serves public/ and api/ separately in production).

Usage (from this directory):
    python3 scripts/local_dev_server.py
"""

import sys
from pathlib import Path

from flask import send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.recommend import app  # noqa: E402

PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"


@app.route("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(PUBLIC_DIR, filename)


if __name__ == "__main__":
    app.run(port=5055, debug=False)
