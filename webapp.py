import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.recommender import SongRecommender


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web"
DATASET_PATH = PROJECT_ROOT / "data" / "spotify_tracks.csv"
HOST = "127.0.0.1"
PORT = 8000


if not DATASET_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found: data/spotify_tracks.csv. Place your Kaggle CSV there before running webapp.py."
    )

recommender = SongRecommender(str(DATASET_PATH))


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/api/recommend":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "Invalid JSON payload"})
            return

        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", 5))
        top_k = max(3, min(10, top_k))
        exclude_reference_artist = bool(payload.get("exclude_reference_artist", True))

        if not query:
            self._send_json(400, {"error": "Query is required"})
            return

        results = recommender.recommend(
            query=query,
            top_k=top_k,
            exclude_reference_artist=exclude_reference_artist,
        )

        response = {
            "source": os.path.basename(DATASET_PATH),
            "mode": recommender.mode,
            "results": [
                {
                    "title": rec.title,
                    "artist": rec.artist,
                    "genre": rec.genre,
                    "score": rec.score,
                    "reason": rec.reason,
                }
                for rec in results
            ],
        }
        self._send_json(200, response)

    def _send_json(self, status_code: int, payload: dict):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"SongSuggest AI web server running at http://{HOST}:{PORT}")
    server.serve_forever()
