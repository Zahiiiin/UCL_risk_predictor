"""
Compatibility file.

The first version of this site started with the command
    gunicorn ucl_predictor_app:server
Hosting services remember that command, so this small file keeps it working.
The real web app is in app.py.
"""
import os

from app import app as server  # noqa: F401  (the name "server" is what the old command looks for)

app = server

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
