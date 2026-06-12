import os

from src.recommender import SongRecommender


SPOTIFY_DATASET = "data/spotify_tracks.csv"
ACTIVE_DATASET = SPOTIFY_DATASET

if not os.path.exists(ACTIVE_DATASET):
    raise FileNotFoundError(
        "Dataset not found: data/spotify_tracks.csv. "
        "Place your Kaggle CSV there before running the app."
    )

recommender = SongRecommender(csv_path=ACTIVE_DATASET)


def recommend_songs(query: str, top_k: int, exclude_reference_artist: bool):
    results = recommender.recommend(
        query=query,
        top_k=top_k,
        exclude_reference_artist=exclude_reference_artist,
    )

    if not results:
        return "Please enter a request, for example: I want nu metal like Korn."

    lines = [
        f"### Recommendations (source: {os.path.basename(ACTIVE_DATASET)}, mode: {recommender.mode})"
    ]
    for idx, rec in enumerate(results, start=1):
        lines.append(
            f"{idx}. **{rec.title}** - {rec.artist} ({rec.genre})  "+
            f"Score: `{rec.score:.3f}`  "+
            f"Why: {rec.reason}"
        )
    return "\n\n".join(lines)


def run_cli() -> None:
    print("SongSuggest AI (CLI mode)")
    print(f"Source: {os.path.basename(ACTIVE_DATASET)} | Mode: {recommender.mode}")
    print("Type your request and press Enter. Type 'exit' to stop.\n")
    while True:
        query = input("Your request> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        print(recommend_songs(query, top_k=5, exclude_reference_artist=True))
        print()


if __name__ == "__main__":
    run_cli()
