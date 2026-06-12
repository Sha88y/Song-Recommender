# How SongSuggest AI Works

This document explains the technical flow of the app so you can present and defend it during your exam.

## 1. Goal of the Application
SongSuggest AI converts a natural-language music request into a ranked list of song recommendations with short explanations.

Examples of requests:
- I want nu metal in the style of Korn
- Give me dark melodic alternative metal
- songs that sound like would? from Alice in Chains

## 2. Main Components
- Web/API layer: [webapp.py](../webapp.py)
- Recommender engine: [src/recommender.py](../src/recommender.py)
- Frontend UI: [web/index.html](../web/index.html)
- Documentation and setup: [README.md](../README.md)

## 3. Data Layer
The app does not use SQL. It uses CSV files from the Spotify/Kaggle dataset:
- [data/spotify_tracks.csv](../data/spotify_tracks.csv): main track table
- [data/data_by_artist.csv](../data/data_by_artist.csv): artist-level aggregate features
- [data/data_by_genres.csv](../data/data_by_genres.csv): genre-level feature prototypes
- [data/data_by_year.csv](../data/data_by_year.csv): year-level context
- [data/data_w_genres.csv](../data/data_w_genres.csv): artist-to-genre mappings

At startup, these files are loaded into memory and transformed into lookup maps for fast ranking.

## 4. Request Flow (End-to-End)
1. User enters a query in the web UI.
2. Frontend sends POST request to /api/recommend.
3. API calls SongRecommender.recommend(...).
4. Recommender parses query intent (genre/artist/track/mood cues).
5. Recommender scores all candidate tracks.
6. Top-K results are returned with title, artist, genre, score, and reason.
7. Frontend renders cards and explanations.

## 5. AI Logic in the Recommender
The app uses a hybrid content-based recommendation approach.

### 5.1 Query understanding
The recommender extracts:
- Explicit genre hints (example: grunge, alternative metal)
- Artist hints (example: style of Korn)
- Reference track hints (example: songs like ...)
- Mood hints (example: chill, sad, upbeat)

### 5.2 Feature profile matching
For each query, it builds a target profile over audio features, such as:
- valence
- energy
- danceability
- acousticness
- tempo
- loudness
- speechiness
- instrumentalness
- year
- popularity

Then it compares this query profile with each track profile.

### 5.3 Style and genre alignment
The score is adjusted using:
- Artist inferred genres
- Genre prototype overlap
- Style tag overlaps
- Penalties when strict query intent is not matched

### 5.4 Reference-track safeguards
For songs like X style prompts, the app adds quality controls:
- Optionally exclude the reference artist
- Penalize weak year alignment
- Penalize low-confidence/very low popularity matches
- Boost title similarity when useful

### 5.5 Diversity pass
After scoring, the app runs a diversity step to avoid returning only one artist in the top results.

## 6. Lexical vs Embedding Mode
The app has two runtime modes:
- lexical: explicit query parsing + feature/style scoring
- embedding: semantic similarity using Sentence Transformers

Model used in embedding mode:
- all-MiniLM-L6-v2

Routing behavior:
- Specific requests (clear artist/genre/track) stay on lexical path.
- Vague natural language can route to embedding path.

Note:
- Embedding mode is optional and only active when dependencies are installed.

## 7. Why This Is AI (for exam explanation)
You can explain the AI component as:
- A content-based recommendation system that combines structured feature similarity with semantic language understanding.
- It performs automated query interpretation and multi-signal ranking.
- It outputs explainable recommendations (reason text per result).

## 8. Deployment and Runtime
You can run the app in multiple ways:
- Local Python run
- VM run
- Docker run with detached mode

Current project also supports environment-based host/port configuration via .env variables for easier local vs VM behavior.

## 9. Limitations (important to mention in exam)
- Recommendations are limited to tracks that exist in the dataset.
- Not all music in the world is covered.
- Quality depends on dataset completeness and metadata quality.
- Some niche prompts can still produce weaker matches.

## 10. Future Improvements
- Add an evaluation benchmark set for repeatable quality testing.
- Add feedback loop (thumbs up/down) for reranking.
- Add explicit filters (year range, popularity range, language).
- Improve handling of rare and ambiguous prompts.

## 11. 30-Second Oral Summary
SongSuggest AI is a hybrid content-based music recommender. It takes natural-language user input, extracts intent signals like genre, artist, mood, and reference tracks, then ranks songs from a Spotify/Kaggle dataset using feature similarity, style overlap, and optional embedding-based semantic matching. The result is shown in a web interface with explanation text per recommendation.
