# Project Blueprint

## Project
SongSuggest AI - text-driven music recommendation system.

## Chosen Direction
- Technical path: recommendation systems with optional pretrained embeddings.
- Domain: music discovery and personalization.
- Route: hybrid content-based ranking (lexical + feature similarity, with optional embedding fallback).

## Core Formula
User query -> query understanding -> ranking over Spotify dataset -> top-k songs + reason per song -> web interface/API

## What Is Implemented Now
- Free-text queries (genre, artist, track, vibe)
- Top-k ranked recommendations
- Reason strings per recommendation
- Web app interface + local API
- CLI fallback mode
- Optional embedding mode when dependencies are available
- Automatic routing between strict lexical path and embedding path

## Main Stack
- Python
- Local web server with `http.server`
- Frontend in HTML/CSS/JavaScript
- Recommender engine in `src/recommender.py`
- Optional ML layer:
  - Sentence Transformers (`all-MiniLM-L6-v2`)
  - NumPy

## Data Layer
CSV files in `data/` (no SQL database):
- `spotify_tracks.csv` (main track-level source)
- `data_by_artist.csv` (artist-level aggregates)
- `data_by_genres.csv` (genre-level feature prototypes)
- `data_by_year.csv` (year-level feature context)
- `data_w_genres.csv` (artist-to-genre mapping)

## AI Logic and Data Flow
1. Parse query intent:
	- explicit genres/subgenres
	- artist hints
	- track hints (for prompts like "songs like ...")
	- mood and style cues
2. Build a target profile and compare with candidate song features.
3. Compute ranking score from multiple signals:
	- lexical/style overlap
	- audio-feature similarity
	- genre/artist/year context
	- relevance penalties/boosts
4. Apply diversity pass and return top-k with explanations.
5. If embedding mode is available and query is vague, optionally use semantic matching.

## Interface and Demo Flow
- Start server: `python webapp.py`
- Open browser: `http://127.0.0.1:8000`
- Enter natural-language request and receive ranked results with reasons

## Scope and Boundaries
- Focus is on one clear problem: text-to-song recommendation.
- Local prototype for demo and oral defense (not production deployment).
- No user accounts, no persistent feedback learning loop yet.

## Known Limitations
- Quality depends on dataset coverage and metadata quality.
- Some niche prompts can still produce weaker matches.
- Embedding mode requires extra dependencies and suitable environment setup.

## Next Improvements (After MVP)
- Add lightweight evaluation set for repeatable quality checks.
- Improve reranking for strict "songs like X" prompts.
- Add explicit query filters (year range, popularity range, language).
- Add a small feedback signal (thumbs up/down) for future reranking.
