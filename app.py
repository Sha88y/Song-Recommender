import os
import importlib

from src.recommender import SongRecommender


SPOTIFY_DATASET = "data/spotify_tracks.csv"
ACTIVE_DATASET = SPOTIFY_DATASET

if not os.path.exists(ACTIVE_DATASET):
    raise FileNotFoundError(
        "Dataset not found: data/spotify_tracks.csv. "
        "Place your Kaggle CSV there before running the app."
    )

recommender = SongRecommender(csv_path=ACTIVE_DATASET)

gr = None
try:
    gr = importlib.import_module("gradio")
except ModuleNotFoundError:
    gr = None


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


def build_gradio_app():
    return gr.Interface(
        fn=recommend_songs,
        inputs=[
            gr.Textbox(
                label="What do you want to listen to?",
                placeholder="Example: I want metal in the style of Korn",
                lines=2,
            ),
            gr.Slider(3, 10, value=5, step=1, label="How many suggestions?"),
            gr.Checkbox(value=True, label="Exclude the exact reference artist from results"),
        ],
        outputs=gr.Markdown(label="AI Suggestions"),
        title="SongSuggest AI",
        description=(
            "Content-based song recommender on Kaggle Spotify data. "
            "Uses lexical mode by default and switches to embedding mode when sentence-transformers is installed. "
            "Dataset path: data/spotify_tracks.csv"
        ),
        examples=[
            ["I want nu metal in the style of Korn", 5, True],
            ["Give me dark melodic alternative metal", 5, False],
            ["I want classic heavy metal with fast riffs", 5, False],
        ],
    )


if __name__ == "__main__":
    if gr is None:
        run_cli()
    else:
        demo = build_gradio_app()
        demo.launch()
