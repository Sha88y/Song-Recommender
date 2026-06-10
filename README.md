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
   - `python -m venv .venv`
2. Activate environment:
   - `\.venv\Scripts\Activate.ps1`

No mandatory package install is needed for CLI mode.

## Run
- `python webapp.py`

Open `http://127.0.0.1:8000` in your browser.

Alternative run mode:
- `python app.py` (CLI by default, Gradio only if installed)

## Use Kaggle Dataset
1. Download the Kaggle Spotify archive zip.
2. Place it at `data/archive.zip`.
3. Extract all CSV files into `data/`.
4. Ensure `data/spotify_tracks.csv` exists (base track file).
5. Start the app with `python webapp.py`.

## Example Inputs
- `I want nu metal in the style of Korn`
- `Give me dark melodic alternative metal`
- `I want classic heavy metal with fast riffs`

## Notes
- This MVP uses multiple Kaggle tables together (tracks + artist + genre + year).
- Base setup runs in lexical mode (no heavy ML dependencies required).

## Which AI Is Used?
The app uses a content-based recommender in `src/recommender.py`.

There are 2 runtime modes:
- `lexical` mode (default): no heavy ML packages required.
- `embedding` mode (optional): enabled automatically when `sentence-transformers` and `numpy` are installed.

Routing behavior:
- The app automatically chooses the recommendation path per query.
- Specific queries (explicit track/artist/genre intent) stay on strict feature + style logic.
- Vague natural-language requests can switch to embedding-based semantic matching when embedding dependencies are available.

When embedding mode is available, the model used is:
- `all-MiniLM-L6-v2`

Important:
- This project does not call OpenAI/ChatGPT at runtime for recommendations.
- Recommendations are computed locally from the Spotify/Kaggle dataset.

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

## Optional Upgrade (VM)
Install embedding dependencies when your VM environment is ready:
- `pip install gradio sentence-transformers numpy`

Then restart the app. It will automatically switch from `lexical` mode to `embedding` mode.
