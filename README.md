# SongSuggest AI

SongSuggest AI is a text-driven music recommendation app.

You describe what you want to hear (for example: "I want nu metal in the style of Korn"), and the app returns ranked song suggestions with short explanations.

## Stack
- Python
- Built-in CLI (works without extra packages)
- Optional: Gradio (web UI)
- Optional: Sentence Transformers + NumPy (embedding mode)

## Project Structure
- `app.py`: CLI and optional Gradio launcher
- `webapp.py`: dependency-free web server
- `web/index.html`: simple web interface
- `src/recommender.py`: recommendation logic
- `data/spotify_tracks.csv`: base track dataset used by the app
- `data/data_by_artist.csv`: artist-level aggregates (used for enrichment)
- `data/data_by_genres.csv`: genre-level feature profiles (used for subgenre alignment)
- `data/data_by_year.csv`: year-level feature profiles (used as fallback context)
- `data/data_w_genres.csv`: artist-to-genre mapping (used for inferred genres)
- `docs/project-blueprint.md`: chosen direction and MVP scope

## Setup (Windows)
1. Create virtual environment:
   - `python -m venv .venv-cpython312`
2. Activate environment:
   - `\.venv-cpython312\Scripts\Activate.ps1`
3. Install dependencies (full setup, including embedding mode):
   - `pip install -r requirements.txt`

Minimal note:
- If you want lexical-only behavior, the app can run without these extra packages.

## Run
- `python webapp.py`

Open `http://127.0.0.1:8000` in your browser.

Optional environment config:
- Copy `.env.example` to `.env`
- Set `SONGSUGGEST_HOST`, `SONGSUGGEST_PORT`, and `SONGSUGGEST_ACCESS_HOST` when needed
- The app auto-loads `.env` on startup

Alternative run mode:
- `python app.py` (CLI by default, Gradio only if installed)

## Run with Docker on VM
1. Build and start:
   - `docker compose up --build -d`
2. Open:
   - `http://<VM-IP>:8000`
3. View logs:
   - `docker compose logs -f`
4. Stop:
   - `docker compose down`

Notes:
- The container exposes port `8000` and binds the app to `0.0.0.0`.
- Make sure VM firewall/security rules allow inbound TCP on `8000`.

## Use Kaggle Dataset
1. Download the Kaggle Spotify archive zip.
2. Place it at `data/archive.zip`.
3. Extract all CSV files into `data/`.
4. Ensure `data/spotify_tracks.csv` exists (base track file).
5. Start the app with `python webapp.py`.

## Example Inputs
- `I want nu metal in the style of Korn`
- `Give me dark melodic alternative metal`
- `give me some chill indie rock`
- `songs that sound like would? from Alice in Chains`

## Notes
- This MVP uses multiple Kaggle tables together (tracks + artist + genre + year).
- Base setup runs in lexical mode (no heavy ML dependencies required).
- Not every song/artist can be requested or returned: recommendations are limited to what exists in the included dataset.
- Expanding to near-complete music coverage would require a much larger dataset and heavier storage/compute.

## Which AI Is Used?
The app uses a content-based recommender in `src/recommender.py` and uses 2 main approaches to compute recommendations:


There are 2 runtime modes:
- `lexical` works purely from explicit query understanding and feature similarity logic.
- `embedding` works from semantic matching of query and track profiles in a vector space (when embedding dependencies are available).

Routing behavior:
- The app automatically chooses the recommendation path per query.
- Specific queries (explicit track/artist/genre intent) stay on strict feature + style logic (lexical).
- Vague natural-language requests can switch to embedding-based semantic matching when embedding dependencies are available. (embedding)
- This works depending on the input. For example, "I want upbeat pop rock" is more likely to trigger the embedding path, while "songs like Smells Like Teen Spirit" is more likely to stay in lexical mode.

When embedding mode is available, the model used is:
- `all-MiniLM-L6-v2`

Important:
- This project does not call OpenAI/ChatGPT at runtime for recommendations.
- Recommendations are computed locally from the Spotify/Kaggle dataset. The app is only able to give suggestions based on what is in the dataset, so it may not know about very recent releases or obscure tracks.
- 

## How The "Database" Is Implemented
This project does not use a SQL database.

Instead, it uses CSV files in `data/` as a data layer:
- `spotify_tracks.csv`: main track-level table
- `data_w_genres.csv`: artist -> genre mapping
- `data_by_artist.csv`: artist-level audio-feature aggregates
- `data_by_genres.csv`: genre-level feature prototypes
- `data_by_year.csv`: year-level feature aggregates

On startup, the recommender loads these files into memory and builds lookup maps (artist, genre, year, and track-title hints) for fast ranking.

## What The AI Looks At During Search
For each user query, the recommender combines multiple signals:

1. Query understanding
- Extracts explicit genres/subgenres from the text.
- Detects mood hints (for example: chill, sad, upbeat).
- Detects artist hints and track-title hints (for "songs like ...").

2. Audio-feature similarity
- Compares query profile vs track profile on features such as:
   `valence`, `energy`, `danceability`, `acousticness`, `tempo`, `loudness`, `speechiness`, `instrumentalness`, `year`, `popularity`, etc.

3. Style and subgenre alignment
- Uses inferred artist genres and genre prototypes.
- Adds boosts/penalties when track style tags overlap (or conflict) with query targets.

4. Quality and relevance safeguards (especially for "songs like X")
- Optionally excludes the exact reference artist.
- Penalizes very weak year alignment to the reference track.
- Penalizes very low-popularity tracks to reduce noisy/obscure matches.

5. Diversity pass
- First pass prefers artist diversity so top results are not all from one artist.

Each result includes a short reason so you can see why it was selected.

## How the lexical route works
1. Parse query for explicit intent (genres, mood, artist hints).
2. Build a target profile from the query (for example: target valence/energy/d
3. anceability based on genre/mood cues).
4. Compare the target profile with candidate tracks from the dataset.
5. Compute a ranking score from multiple signals (lexical/style overlap, audio-feature similarity, genre/artist/year context).
6. Apply a diversity pass and return the top-k results with explanations.

## How the embedding route works
1. Encode the user query into an embedding vector using the Sentence Transformers model.
2. Precompute track embeddings from the dataset (done at startup).
3. Compute cosine similarity between the query embedding and track embeddings.
4. Combine embedding similarity with other signals (for example: genre/artist hints) to compute
5. a final ranking score.
6. Return the top-k results with explanations.

 
# Conclusion
This project is a focused implementation of a text-driven music recommendation system using a hybrid content-based approach. It combines lexical query understanding with audio-feature similarity and optional embedding-based semantic matching to deliver personalized song suggestions from the Spotify dataset. The app runs locally with minimal dependencies, making it accessible while still demonstrating core AI concepts in recommendation systems.
