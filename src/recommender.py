import csv
import importlib
import math
import re
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Recommendation:
    """Final recommendation payload returned to CLI/API/UI layers."""

    title: str
    artist: str
    genre: str
    score: float
    reason: str


class SongRecommender:
    """Hybrid recommender combining lexical, feature-based, and optional embedding scoring."""

    # Numeric track attributes used for profile building and similarity computation.
    NUMERIC_FEATURES = [
        "valence",
        "year",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "explicit",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "popularity",
        "speechiness",
        "tempo",
    ]

    STOPWORDS = {
        "i",
        "want",
        "like",
        "the",
        "a",
        "an",
        "to",
        "in",
        "of",
        "and",
        "with",
        "for",
        "now",
        "genre",
        "style",
        "song",
        "songs",
        "music",
        "me",
        "give",
        "this",
        "that",
    }

    MOOD_HINTS = {
        "chill": {"energy": 0.35, "tempo": 95.0, "valence": 0.45},
        "calm": {"energy": 0.25, "tempo": 88.0, "valence": 0.45},
        "sad": {"valence": 0.25, "energy": 0.35, "tempo": 92.0},
        "happy": {"valence": 0.82, "energy": 0.68, "tempo": 122.0},
        "party": {"danceability": 0.82, "energy": 0.8, "tempo": 126.0},
        "upbeat": {"energy": 0.76, "tempo": 124.0},
    }

    GENRE_EXPANSION_TOP_K = 4
    TRACK_YEAR_SOFT_TOLERANCE = 8
    TRACK_YEAR_MAX_PENALTY = 0.16
    TRACK_LOW_POPULARITY_THRESHOLD = 20.0
    TRACK_LOW_POPULARITY_MAX_PENALTY = 0.08

    BROAD_GENRE_TAGS = {
        "rock",
        "metal",
        "pop",
        "jazz",
        "rap",
        "hip hop",
        "country",
        "blues",
        "soul",
        "dance",
        "electronic",
        "hard rock",
    }

    NATURAL_LANGUAGE_HINTS = {
        "vibe",
        "vibes",
        "feeling",
        "feel",
        "mood",
        "atmospheric",
        "dreamy",
        "chill",
        "calm",
        "sad",
        "happy",
        "upbeat",
        "party",
        "ontspannen",
        "rustig",
        "somber",
        "energie",
    }

    def __init__(self, csv_path: str, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load datasets, precompute lookup maps, and initialize optional ML dependencies."""

        self.data_dir = Path(csv_path).resolve().parent
        # Dataset-derived metadata maps used during query understanding and scoring.
        self.artist_genre_map = self._load_artist_genre_map()
        self.artist_feature_map = self._load_artist_feature_map()
        self.artist_keys_sorted = sorted(self.artist_feature_map.keys(), key=len, reverse=True)
        self.genre_feature_prototypes = self._load_genre_feature_prototypes()
        self.dataset_genre_keywords = sorted(self.genre_feature_prototypes.keys(), key=len, reverse=True)
        self.year_feature_map = self._load_year_feature_map()

        # Main song rows used as candidate pool.
        self.rows = self._load_rows(csv_path)
        if not self.rows:
            raise ValueError("Dataset is empty or could not be parsed.")
        self._attach_style_tags()
        self.track_seed_map = self._build_track_seed_map()
        self.track_keys_sorted = sorted(self.track_seed_map.keys(), key=len, reverse=True)
        self.feature_stats = self._compute_feature_stats()
        self.dataset_mean_profile = {feature: self.feature_stats[feature][0] for feature in self.NUMERIC_FEATURES}

        self.np: Optional[Any] = None
        self._sentence_transformer_cls: Optional[Any] = None
        self._load_optional_dependencies()

        self.mode = "lexical"
        self.model = None
        self.song_embeddings = None
        self.embedding_index_ready = False
        # Embedding mode is optional and enabled only when dependencies are available.
        if self._sentence_transformer_cls is not None and self.np is not None:
            self.model = self._sentence_transformer_cls(model_name)
            self.mode = "embedding"

    def _ensure_song_embeddings(self) -> bool:
        """Build song embedding index on demand to keep startup responsive."""

        if self.embedding_index_ready and self.song_embeddings is not None:
            return True
        if self.model is None or self.np is None:
            return False

        # Build embedding index lazily to avoid long startup stalls.
        self.song_embeddings = self.model.encode(
            [row["search_text"] for row in self.rows],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        self.embedding_index_ready = True
        return True

    def _load_optional_dependencies(self) -> None:
        """Try importing optional dependencies without hard-failing lexical mode."""

        try:
            self.np = importlib.import_module("numpy")
        except ModuleNotFoundError:
            self.np = None

        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._sentence_transformer_cls = getattr(sentence_transformers, "SentenceTransformer", None)
        except ModuleNotFoundError:
            self._sentence_transformer_cls = None

    @staticmethod
    def _get_value(raw_row: Dict[str, str], candidates: List[str], default: str = "") -> str:
        """Return first non-empty value from possible column names."""

        for key in candidates:
            value = raw_row.get(key)
            if value is not None and str(value).strip() != "":
                return str(value).strip()
        return default

    @staticmethod
    def _clean_artist(value: str) -> str:
        return value.replace("[", "").replace("]", "").replace("'", "").strip()

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Convert values safely to float, returning None for missing/invalid inputs."""

        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_list_like(value: str) -> List[str]:
        """Parse list-like strings from CSV fields (supports literal list or comma-separated text)."""

        text = (value or "").strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = literal_eval(text)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            except Exception:
                pass
        return [item.strip() for item in text.split(",") if item.strip()]

    @staticmethod
    def _artist_key(name: str) -> str:
        lowered = name.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
        return lowered.strip()

    @staticmethod
    def _norm_tag(value: str) -> str:
        return re.sub(r"\s+", " ", value.lower().strip())

    @staticmethod
    def _is_missing_label(value: str) -> bool:
        normalized = (value or "").strip().lower()
        return normalized in {"", "unknown", "none", "nan", "n/a"}

    @staticmethod
    def _title_key(value: str) -> str:
        lowered = value.lower().strip()
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered.strip()

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        """Boundary-aware phrase matching to avoid partial-token false positives."""

        if not text or not phrase:
            return False
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        return re.search(pattern, text) is not None

    def _load_artist_genre_map(self) -> Dict[str, set]:
        """Load artist -> genres mapping from dataset enrichment file."""

        path = self.data_dir / "data_w_genres.csv"
        mapping: Dict[str, set] = {}
        if not path.exists():
            return mapping

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                artists = self._parse_list_like(str(row.get("artists", "")))
                genres = [self._norm_tag(g) for g in self._parse_list_like(str(row.get("genres", "")))]
                if not artists or not genres:
                    continue
                for artist in artists:
                    key = self._artist_key(artist)
                    if not key:
                        continue
                    if key not in mapping:
                        mapping[key] = set()
                    mapping[key].update(genres)
        return mapping

    def _load_genre_feature_prototypes(self) -> Dict[str, Dict[str, float]]:
        """Load average feature profiles per genre for strict style alignment."""

        path = self.data_dir / "data_by_genres.csv"
        prototypes: Dict[str, Dict[str, float]] = {}
        if not path.exists():
            return prototypes

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                genre_name = self._norm_tag(str(row.get("genres", "")))
                if not genre_name:
                    continue
                feature_profile: Dict[str, float] = {}
                for feature in self.NUMERIC_FEATURES:
                    value = self._safe_float(row.get(feature))
                    if value is not None:
                        feature_profile[feature] = value
                if feature_profile:
                    prototypes[genre_name] = feature_profile
        return prototypes

    def _load_artist_feature_map(self) -> Dict[str, Dict[str, float]]:
        """Load artist-level feature profiles used for artist-seeded recommendations."""

        path = self.data_dir / "data_by_artist.csv"
        mapping: Dict[str, Dict[str, float]] = {}
        if not path.exists():
            return mapping

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                artist_key = self._artist_key(str(row.get("artists", "")))
                if not artist_key:
                    continue
                feature_profile: Dict[str, float] = {}
                for feature in self.NUMERIC_FEATURES:
                    value = self._safe_float(row.get(feature))
                    if value is not None:
                        feature_profile[feature] = value
                if feature_profile:
                    mapping[artist_key] = feature_profile
        return mapping

    def _load_year_feature_map(self) -> Dict[int, Dict[str, float]]:
        """Load year-level fallback feature profiles used when track rows miss values."""

        path = self.data_dir / "data_by_year.csv"
        mapping: Dict[int, Dict[str, float]] = {}
        if not path.exists():
            return mapping

        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                year_value = self._safe_float(row.get("year"))
                if year_value is None:
                    continue
                year = int(year_value)
                feature_profile: Dict[str, float] = {}
                for feature in self.NUMERIC_FEATURES:
                    value = self._safe_float(row.get(feature))
                    if value is not None:
                        feature_profile[feature] = value
                if feature_profile:
                    mapping[year] = feature_profile
        return mapping

    def _expand_genre_tags(self, tags: List[str]) -> List[str]:
        """Expand genre tags by token-overlap with known dataset genres."""

        expanded = list(tags)
        for tag in tags:
            source_tokens = {token for token in self._tokenize(tag) if token not in self.BROAD_GENRE_TAGS}
            if not source_tokens:
                continue
            scored: List[tuple] = []
            for genre in self.dataset_genre_keywords:
                if genre == tag:
                    continue
                genre_tokens = set(self._tokenize(genre))
                overlap = source_tokens.intersection(genre_tokens)
                if not overlap:
                    continue
                score = len(overlap) / max(len(source_tokens), 1)
                if score >= 0.5:
                    scored.append((genre, score))
            scored.sort(key=lambda item: (item[1], -len(item[0])), reverse=True)
            expanded.extend([genre for genre, _score in scored[: self.GENRE_EXPANSION_TOP_K]])
        deduped: List[str] = []
        for tag in expanded:
            if tag not in deduped:
                deduped.append(tag)
        return deduped

    def _normalize_row(self, raw_row: Dict[str, str]) -> Dict[str, str]:
        """Normalize a raw CSV row into a consistent internal schema."""

        title = self._get_value(raw_row, ["title", "track_name", "name", "song", "song_name"], "unknown title")
        artist = self._clean_artist(self._get_value(raw_row, ["artist", "artists", "artist_name"], "unknown artist"))
        genre = self._get_value(raw_row, ["genre", "track_genre", "playlist_genre"], "unknown")
        tags = self._get_value(raw_row, ["tags", "playlist_subgenre", "subgenre"])

        # Prefer a meaningful genre label when source rows omit explicit genre columns.
        if self._is_missing_label(genre):
            parsed_tags = [self._norm_tag(tag) for tag in self._parse_list_like(tags)]
            parsed_tags = [tag for tag in parsed_tags if not self._is_missing_label(tag)]
            if parsed_tags:
                genre = parsed_tags[0]
            else:
                inferred_from_artist = sorted(self.artist_genre_map.get(self._artist_key(artist), set()))
                if inferred_from_artist:
                    genre = inferred_from_artist[0]
                else:
                    genre = "unknown"

        metrics = []
        for key in ["popularity", "energy", "valence", "danceability", "tempo", "year"]:
            value = raw_row.get(key)
            if value is not None and str(value).strip() != "":
                metrics.append(f"{key} {value}")

        description = " ".join(metrics).strip()
        if not description:
            description = self._get_value(raw_row, ["description", "album_name"], "")

        normalized = {
            "title": title,
            "artist": artist,
            "genre": genre,
            "tags": tags,
            "description": description,
            "inferred_genres": [],
        }
        for feature in self.NUMERIC_FEATURES:
            normalized[feature] = self._safe_float(raw_row.get(feature))

        year_val = normalized.get("year")
        if year_val is not None:
            # Fill missing numeric features with year-level averages when available.
            year_profile = self.year_feature_map.get(int(year_val))
            if year_profile:
                for feature, value in year_profile.items():
                    if normalized.get(feature) is None:
                        normalized[feature] = value

        normalized["search_text"] = self._build_search_text(normalized)
        return normalized

    def _compute_feature_stats(self) -> Dict[str, tuple]:
        """Compute mean/std per feature for standardized cosine similarity."""

        stats: Dict[str, tuple] = {}
        for feature in self.NUMERIC_FEATURES:
            values = [row[feature] for row in self.rows if row.get(feature) is not None]
            if not values:
                stats[feature] = (0.0, 1.0)
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
            std = math.sqrt(variance) if variance > 0 else 1.0
            stats[feature] = (mean, std)
        return stats

    def _load_rows(self, csv_path: str) -> List[Dict[str, str]]:
        """Load and normalize all candidate tracks from the base CSV."""

        rows: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                rows.append(self._normalize_row(raw_row))
        return rows

    def _style_tags_for_artist(self, artist: str) -> List[str]:
        """Get inferred style tags (genres) for an artist."""

        tags: List[str] = []

        artist_key = self._artist_key(artist)
        inferred_genres = self.artist_genre_map.get(artist_key, set())
        for genre in inferred_genres:
            tags.append(genre)

        deduped: List[str] = []
        for tag in tags:
            if tag not in deduped:
                deduped.append(tag)
        return deduped

    def _filter_style_tags_for_row(self, style_tags: List[str], row: Dict[str, Any]) -> List[str]:
        """Keep only style tags that match row context tokens when possible."""

        context_text = f"{row.get('genre', '')} {row.get('tags', '')}".strip()
        context_tokens = set(self._tokenize(context_text))
        if not context_tokens:
            return style_tags

        filtered: List[str] = []
        for tag in style_tags:
            tag_tokens = set(self._tokenize(tag))
            if tag_tokens.intersection(context_tokens):
                filtered.append(tag)

        return filtered if filtered else style_tags

    def _attach_style_tags(self) -> None:
        """Attach inferred style tags to each row for faster scoring."""

        for row in self.rows:
            style_tags = self._style_tags_for_artist(row.get("artist", ""))
            style_tags = self._filter_style_tags_for_row(style_tags, row)
            row["style_tags"] = style_tags
            row["inferred_genres"] = [tag for tag in style_tags if tag in self.genre_feature_prototypes]

    def _build_track_seed_map(self) -> Dict[str, Dict[str, Any]]:
        """Create title-keyed representative tracks used for 'songs like X' hints."""

        representatives: Dict[str, Dict[str, Any]] = {}
        for row in self.rows:
            title_key = self._title_key(str(row.get("title", "")))
            if len(title_key) < 6:
                continue

            popularity = row.get("popularity") if row.get("popularity") is not None else 0.0
            current = representatives.get(title_key)
            if current is None or float(popularity) > float(current.get("popularity", -1.0)):
                representatives[title_key] = {
                    "row": row,
                    "popularity": float(popularity),
                }

        seed_map: Dict[str, Dict[str, Any]] = {}
        for title_key, entry in representatives.items():
            representative = entry["row"]
            profile: Dict[str, float] = {}
            for feature in self.NUMERIC_FEATURES:
                value = representative.get(feature)
                if value is not None:
                    profile[feature] = float(value)
            if not profile:
                continue

            seed_map[title_key] = {
                "profile": profile,
                "title": representative.get("title", ""),
                "artist": representative.get("artist", ""),
                "genre": representative.get("genre", ""),
                "style_tags": list(representative.get("style_tags", [])),
                "inferred_genres": list(representative.get("inferred_genres", [])),
            }

        return seed_map

    def _extract_track_hint(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect whether a known track title appears in the query."""

        normalized_query = self._title_key(query)
        for title_key in self.track_keys_sorted:
            if len(title_key) < 8:
                continue
            if self._contains_phrase(normalized_query, title_key):
                return self.track_seed_map[title_key]
        return None

    def _track_reference_year(self, track_hint: Optional[Dict[str, Any]]) -> Optional[int]:
        """Extract reference track year for era-alignment penalties/boosts."""

        if not track_hint:
            return None
        profile = track_hint.get("profile")
        if not isinstance(profile, dict):
            return None
        value = profile.get("year")
        if value is None:
            return None
        try:
            return int(round(float(value)))
        except Exception:
            return None

    @staticmethod
    def _build_search_text(row: Dict[str, str]) -> str:
        return " ".join(
            [
                str(row.get("title", "")),
                str(row.get("artist", "")),
                str(row.get("genre", "")),
                str(row.get("tags", "")),
                str(row.get("description", "")),
            ]
        ).strip()

    @staticmethod
    def _extract_artist_hint_from_patterns(query: str) -> Optional[str]:
        """Fallback artist-hint extraction from explicit query phrases."""

        patterns = [
            r"(?:like|zoals)\s+([a-zA-Z0-9\s]+)$",
            r"(?:style of|stijl van|genre van)\s+([a-zA-Z0-9\s]+)$",
            r"(?:similar to)\s+([a-zA-Z0-9\s]+)$",
        ]
        q = query.strip().lower()
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return match.group(1).strip()
        return None

    def _extract_artist_hint(self, query: str) -> Optional[str]:
        """Extract best artist hint using known artist keys, then fallback patterns."""

        q_normalized = self._artist_key(query)

        # Prefer artists known from dataset-derived artist profiles.
        for artist_key in self.artist_keys_sorted:
            if len(artist_key) < 4:
                continue
            if artist_key and self._contains_phrase(q_normalized, artist_key):
                return artist_key

        return self._extract_artist_hint_from_patterns(query)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token and len(token) >= 3]

    def _extract_query_style_tags(
        self,
        query: str,
        artist_hint: Optional[str],
        track_hint: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Build style tag targets from explicit genres, moods, artist and track context."""

        q_lower = query.lower()
        tags: List[str] = []
        explicit_genres = self._extract_explicit_query_genres(query)

        tags.extend(explicit_genres)

        for mood_key in self.MOOD_HINTS.keys():
            if mood_key in q_lower:
                tags.append(mood_key)

        if artist_hint:
            artist_tags = self.artist_genre_map.get(self._artist_key(artist_hint), set())
            if explicit_genres:
                related_targets = set(explicit_genres)
                related_targets.update(self._expand_genre_tags(explicit_genres))
                artist_tags = {tag for tag in artist_tags if tag in related_targets}
            else:
                # Avoid over-broad labels dominating strict matching.
                artist_tags = {tag for tag in artist_tags if tag not in self.BROAD_GENRE_TAGS}
            tags.extend(list(artist_tags))

        if track_hint:
            track_tags = set(track_hint.get("inferred_genres", []))
            if not track_tags:
                track_tags = set(track_hint.get("style_tags", []))
            if explicit_genres:
                related_targets = set(explicit_genres)
                related_targets.update(self._expand_genre_tags(explicit_genres))
                track_tags = {tag for tag in track_tags if tag in related_targets}
            else:
                track_tags = {tag for tag in track_tags if tag not in self.BROAD_GENRE_TAGS}
            tags.extend(list(track_tags))

        # For "songs like <track>" queries without explicit genres, avoid aggressive expansion.
        if track_hint and not explicit_genres:
            expanded_tags = list(tags)
        else:
            expansion_sources = explicit_genres if explicit_genres else tags
            expanded_tags = self._expand_genre_tags(expansion_sources)
            expanded_tags.extend(tags)

        deduped: List[str] = []
        for tag in expanded_tags:
            if tag not in deduped:
                deduped.append(tag)
        return deduped

    def _extract_explicit_query_genres(self, query: str) -> List[str]:
        """Extract explicit genre mentions with boundary-aware matching and pruning."""

        q_lower = query.lower()
        matches: List[str] = []
        for genre_key in self.dataset_genre_keywords:
            if len(genre_key) < 4:
                continue
            if self._contains_phrase(q_lower, genre_key):
                matches.append(genre_key)

        deduped: List[str] = []
        for tag in matches:
            if tag not in deduped:
                deduped.append(tag)

        pruned: List[str] = []
        for tag in deduped:
            if tag in self.BROAD_GENRE_TAGS:
                is_part_of_specific = any(
                    other != tag and self._contains_phrase(other, tag)
                    for other in deduped
                )
                if is_part_of_specific:
                    continue
            pruned.append(tag)
        return pruned

    def _strict_targets_from_query(self, query_style_tags: List[str], explicit_query_genres: Optional[List[str]] = None) -> List[str]:
        """Build strict target genres used to reward/penalize style alignment."""

        if explicit_query_genres:
            source_tags = explicit_query_genres
        else:
            source_tags = query_style_tags

        targets: List[str] = [tag for tag in source_tags if tag in self.genre_feature_prototypes]
        for tag in source_tags:
            for expanded in self._expand_genre_tags([tag]):
                if expanded in self.genre_feature_prototypes:
                    targets.append(expanded)

        deduped: List[str] = []
        for tag in targets:
            if tag not in deduped:
                if explicit_query_genres and tag in self.BROAD_GENRE_TAGS and tag not in explicit_query_genres:
                    continue
                deduped.append(tag)
        return deduped
    # Style-alignment adjustment is capped to avoid dominating relevance signals when strict targets are present.
    @staticmethod
    def _strict_style_adjustment(row_style_tags: List[str], inferred_genres: List[str], strict_targets: List[str]) -> float:
        """Return style-alignment score adjustment relative to strict query targets."""

        if not strict_targets:
            return 0.0
        merged_tags = set(row_style_tags).union(set(inferred_genres))
        if not merged_tags:
            return -0.08
        overlap = merged_tags.intersection(set(strict_targets))
        if overlap:
            return min(0.1, 0.035 * len(overlap))
        return -0.16

    def _lexical_similarity(self, query: str, search_text: str) -> float:
        """Lightweight lexical overlap score between query and row search text."""

        query_tokens = set(token for token in self._tokenize(query) if token not in self.STOPWORDS)
        if not query_tokens:
            return 0.0
        row_tokens = set(self._tokenize(search_text))
        overlap = len(query_tokens.intersection(row_tokens))
        return overlap / max(len(query_tokens), 1)

    def _strip_artist_hint_from_query(self, query: str, artist_hint: Optional[str]) -> str:
        """Remove known artist phrases to avoid over-weighting exact-name token overlap."""

        if not artist_hint:
            cleaned = query.lower()
            cleaned = cleaned.replace("stoner rock", "rock")
            return re.sub(r"\s+", " ", cleaned).strip()

        cleaned = query.lower()
        cleaned = cleaned.replace(artist_hint.lower(), " ")
        cleaned = cleaned.replace("stoner rock", "rock")
        for phrase in ["similar to", "style of", "genre van", "stijl van", "like", "zoals"]:
            cleaned = cleaned.replace(phrase, " ")
        return re.sub(r"\s+", " ", cleaned).strip()

    def _standardized_vector(self, profile: Dict[str, float]) -> List[float]:
        """Transform feature dict into standardized vector using dataset mean/std."""

        vector = []
        for feature in self.NUMERIC_FEATURES:
            mean, std = self.feature_stats.get(feature, (0.0, 1.0))
            value = profile.get(feature, mean)
            vector.append((value - mean) / std if std else 0.0)
        return vector

    @staticmethod
    def _cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _build_artist_seed_profile(self, artist_hint: Optional[str]) -> Optional[Dict[str, float]]:
        """Build artist seed profile from precomputed map or row-level aggregation fallback."""

        if not artist_hint:
            return None

        artist_key = self._artist_key(artist_hint)
        if artist_key in self.artist_feature_map:
            return dict(self.artist_feature_map[artist_key])

        matched_rows = [row for row in self.rows if artist_hint in row["artist"].lower()]
        if not matched_rows:
            return None

        profile: Dict[str, float] = {}
        for feature in self.NUMERIC_FEATURES:
            values = [row[feature] for row in matched_rows if row.get(feature) is not None]
            if values:
                profile[feature] = sum(values) / len(values)
        return profile if profile else None

    def _build_query_profile(
        self,
        query: str,
        artist_hint: Optional[str],
        track_hint: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, float]]:
        """Build query-side numeric profile by combining track/artist seeds and mood hints."""

        profile = dict(self.dataset_mean_profile)
        has_seed = False

        if track_hint and track_hint.get("profile"):
            profile.update(track_hint["profile"])
            has_seed = True

        seed_profile = self._build_artist_seed_profile(artist_hint)
        if seed_profile:
            if has_seed:
                for feature, value in seed_profile.items():
                    base = profile.get(feature, value)
                    profile[feature] = (0.7 * base) + (0.3 * value)
            else:
                profile.update(seed_profile)
            has_seed = True

        applied = 0
        query_lower = query.lower()
        for hint, overrides in self.MOOD_HINTS.items():
            if hint in query_lower:
                applied += 1
                for feature, value in overrides.items():
                    profile[feature] = float(value)

        return profile if has_seed or applied > 0 else None

    def _genre_profile_from_targets(self, strict_targets: List[str]) -> Optional[Dict[str, float]]:
        """Merge feature prototypes for strict target genres into one profile."""

        profiles = [self.genre_feature_prototypes[tag] for tag in strict_targets if tag in self.genre_feature_prototypes]
        if not profiles:
            return None

        merged: Dict[str, float] = {}
        for feature in self.NUMERIC_FEATURES:
            values = [profile[feature] for profile in profiles if feature in profile]
            if values:
                merged[feature] = sum(values) / len(values)
        return merged if merged else None

    def _feature_similarity(
        self,
        query: str,
        artist_hint: Optional[str],
        query_style_tags: List[str],
        track_hint: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[float]]:
        """Compute per-row feature similarity scores in [0, 1] using cosine similarity."""

        profile = self._build_query_profile(query, artist_hint, track_hint)
        explicit_query_genres = self._extract_explicit_query_genres(query)
        strict_targets = self._strict_targets_from_query(query_style_tags, explicit_query_genres)
        genre_profile = self._genre_profile_from_targets(strict_targets)
        if genre_profile:
            if profile is None:
                profile = dict(self.dataset_mean_profile)
            for feature, value in genre_profile.items():
                base = profile.get(feature, self.dataset_mean_profile.get(feature, value))
                profile[feature] = (0.4 * base) + (0.6 * value)

        if profile is None:
            return None

        query_vector = self._standardized_vector(profile)
        similarities: List[float] = []
        for row in self.rows:
            row_profile = {
                feature: (row[feature] if row.get(feature) is not None else self.dataset_mean_profile[feature])
                for feature in self.NUMERIC_FEATURES
            }
            row_vector = self._standardized_vector(row_profile)
            cosine = self._cosine_similarity(query_vector, row_vector)
            # Map cosine from [-1, 1] to [0, 1] for easier blending.
            similarities.append((cosine + 1.0) / 2.0)
        return similarities
    # Heuristic routing to embedding mode for vague queries without specific feature/style hints.
    def _should_use_embedding(
        self,
        query: str,
        explicit_query_genres: List[str],
        artist_hint: Optional[str],
        track_hint: Optional[Dict[str, Any]],
        reference_phrase: Optional[str],
    ) -> bool:
        """Route vague natural-language queries to embedding mode when available."""

        # Keep specific requests on strict feature/style logic.
        if track_hint or artist_hint or explicit_query_genres or reference_phrase:
            return False

        tokens = [token for token in self._tokenize(query) if token not in self.STOPWORDS]
        if len(tokens) <= 3:
            return False

        hint_hits = sum(1 for token in tokens if token in self.NATURAL_LANGUAGE_HINTS)
        return hint_hits > 0 or len(tokens) >= 6

    def _extract_reference_phrase(self, query: str) -> Optional[str]:
        """Extract free-form reference phrase from 'songs like ...' style queries."""

        q = query.lower().strip()
        patterns = [
            r"(?:songs?|tracks?|music|nummers?)\s+(?:like|zoals)\s+(.+)$",
            r"(?:similar to|like|zoals)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            phrase = match.group(1).strip(" \t\n\r\"'.,!?;:-")
            tokens = [t for t in self._tokenize(phrase) if t not in self.STOPWORDS]
            if len(tokens) >= 3:
                return phrase
        return None

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        exclude_reference_artist: bool = False,
    ) -> List[Recommendation]:
        """Main inference entrypoint: parse query, score rows, rerank, and explain results."""

        if not query or not query.strip():
            return []

        artist_hint = self._extract_artist_hint(query)
        track_hint = self._extract_track_hint(query)
        if track_hint and artist_hint:
            track_title_tokens = set(self._tokenize(str(track_hint.get("title", ""))))
            artist_tokens = self._tokenize(artist_hint)
            # Avoid false positives where a single token from the song title is treated as an artist.
            if artist_hint in track_title_tokens:
                artist_hint = None
        track_hint_artist = self._artist_key(str(track_hint.get("artist", ""))) if track_hint else ""
        reference_year = self._track_reference_year(track_hint)
        lexical_query = self._strip_artist_hint_from_query(query, artist_hint)
        explicit_query_genres = self._extract_explicit_query_genres(query)
        reference_phrase = self._extract_reference_phrase(query)
        query_style_tags = self._extract_query_style_tags(query, artist_hint, track_hint)
        strict_targets = self._strict_targets_from_query(query_style_tags, explicit_query_genres)

        embedding_available = self.mode == "embedding" and self.model is not None and self.np is not None
        use_embedding = embedding_available and self._should_use_embedding(
            query=query,
            explicit_query_genres=explicit_query_genres,
            artist_hint=artist_hint,
            track_hint=track_hint,
            reference_phrase=reference_phrase,
        )

        if use_embedding and self._ensure_song_embeddings():
            # Pure semantic query-to-track similarity in embedding space.
            query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            similarities = self.np.dot(self.song_embeddings, query_embedding).tolist()
        else:
            # Hybrid lexical + feature scoring path (default and for specific intents).
            lexical_similarities = [self._lexical_similarity(lexical_query, row["search_text"]) for row in self.rows]
            feature_similarities = self._feature_similarity(query, artist_hint, query_style_tags, track_hint)
            if feature_similarities is not None:
                if track_hint:
                    feature_weight = 0.92
                    lexical_weight = 0.08
                else:
                    feature_weight = 0.85 if strict_targets else 0.7
                    lexical_weight = 0.15 if strict_targets else 0.3
                similarities = [
                    (feature_weight * feature_score) + (lexical_weight * lexical_score)
                    for feature_score, lexical_score in zip(feature_similarities, lexical_similarities)
                ]
            else:
                similarities = lexical_similarities

        q_lower = query.lower()
        query_tokens = q_lower.split()
        reference_tokens = set()
        if reference_phrase:
            reference_tokens = set(token for token in self._tokenize(reference_phrase) if token not in self.STOPWORDS)

        scored_items = []
        for row, base_score in zip(self.rows, similarities):
            # Local score adjustments for better practical recommendation quality.
            genre_lower = row["genre"].lower()
            artist_lower = row["artist"].lower()
            genre_boost = 0.03 if genre_lower and (genre_lower in q_lower or any(token in genre_lower for token in query_tokens)) else 0.0
            artist_boost = 0.02 if artist_hint and artist_hint in artist_lower else 0.0
            row_style_tags = row.get("style_tags", [])
            inferred_genres = row.get("inferred_genres", [])
            style_boost = 0.0
            if query_style_tags and row_style_tags:
                overlap = set(query_style_tags).intersection(set(row_style_tags))
                if overlap:
                    style_boost = min(0.08, 0.025 * len(overlap))

            track_style_boost = 0.0
            if track_hint:
                reference_tags = set(track_hint.get("style_tags", [])).union(set(track_hint.get("inferred_genres", [])))
                if reference_tags:
                    overlap = reference_tags.intersection(set(row_style_tags).union(set(inferred_genres)))
                    if overlap:
                        track_style_boost = min(0.1, 0.03 * len(overlap))

            strict_adjust = self._strict_style_adjustment(row_style_tags, inferred_genres, strict_targets)

            popularity_boost = 0.0
            if track_hint:
                popularity = row.get("popularity") if row.get("popularity") is not None else 0.0
                popularity_boost = 0.03 * max(0.0, min(100.0, float(popularity))) / 100.0

            if track_hint and row.get("title", "").lower() == str(track_hint.get("title", "")).lower():
                continue

            if exclude_reference_artist and track_hint_artist and self._artist_key(artist_lower) == track_hint_artist:
                continue

            if exclude_reference_artist and artist_hint and artist_hint in artist_lower:
                continue

            reference_title_boost = 0.0
            if reference_tokens:
                row_title_tokens = set(self._tokenize(row.get("title", "")))
                overlap = reference_tokens.intersection(row_title_tokens)
                if overlap:
                    overlap_ratio = len(overlap) / max(len(reference_tokens), 1)
                    reference_title_boost = min(0.14, 0.14 * overlap_ratio)

            year_penalty = 0.0
            if track_hint and reference_year is not None and row.get("year") is not None:
                year_diff = abs(int(row["year"]) - reference_year)
                if year_diff > self.TRACK_YEAR_SOFT_TOLERANCE:
                    year_penalty = -min(
                        self.TRACK_YEAR_MAX_PENALTY,
                        0.01 * (year_diff - self.TRACK_YEAR_SOFT_TOLERANCE),
                    )

            popularity_penalty = 0.0
            if track_hint:
                popularity = row.get("popularity") if row.get("popularity") is not None else 0.0
                popularity = float(popularity)
                if popularity < self.TRACK_LOW_POPULARITY_THRESHOLD:
                    ratio = (self.TRACK_LOW_POPULARITY_THRESHOLD - popularity) / self.TRACK_LOW_POPULARITY_THRESHOLD
                    popularity_penalty = -min(self.TRACK_LOW_POPULARITY_MAX_PENALTY, self.TRACK_LOW_POPULARITY_MAX_PENALTY * ratio)

            final_score = float(
                base_score
                + genre_boost
                + artist_boost
                + style_boost
                + track_style_boost
                + strict_adjust
                + popularity_boost
                + reference_title_boost
                + year_penalty
                + popularity_penalty
            )
            scored_items.append(
                (
                    row,
                    genre_boost,
                    artist_boost,
                    style_boost,
                    track_style_boost,
                    strict_adjust,
                    popularity_boost,
                    reference_title_boost,
                    year_penalty,
                    popularity_penalty,
                    final_score,
                )
            )

        ranked_all = sorted(scored_items, key=lambda item: item[10], reverse=True)
        ranked = []
        seen = set()
        selected_artists = set()

        # Pass 1: prioritize artist diversity.
        for item in ranked_all:
            row = item[0]
            key = (row["title"].strip().lower(), row["artist"].strip().lower())
            if key in seen:
                continue
            artist_key = row["artist"].strip().lower()
            if artist_key in selected_artists:
                continue
            seen.add(key)
            selected_artists.add(artist_key)
            ranked.append(item)
            if len(ranked) >= top_k:
                break

        # Pass 2: fill remaining slots by best score.
        if len(ranked) < top_k:
            for item in ranked_all:
                row = item[0]
                key = (row["title"].strip().lower(), row["artist"].strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                ranked.append(item)
                if len(ranked) >= top_k:
                    break

        recommendations: List[Recommendation] = []
        for row, genre_boost, artist_boost, style_boost, track_style_boost, strict_adjust, popularity_boost, reference_title_boost, year_penalty, popularity_penalty, final_score in ranked:
            # Build human-readable reasoning so recommendations are explainable.
            reason_parts = []
            if genre_boost > 0:
                reason_parts.append("genre matches your request")
            if artist_boost > 0:
                reason_parts.append("artist/style hint overlap")
            if style_boost > 0:
                reason_parts.append("artist-style profile similarity")
            if track_style_boost > 0:
                reason_parts.append("similar style tags as reference track")
            if strict_adjust > 0:
                reason_parts.append("subgenre alignment")
            if popularity_boost > 0.015:
                reason_parts.append("strong known-track relevance")
            if reference_title_boost > 0.05:
                reason_parts.append("title similarity with your reference")
            if year_penalty < -0.05:
                reason_parts.append("weaker era alignment")
            if popularity_penalty < -0.03:
                reason_parts.append("lower catalog confidence")
            if not reason_parts:
                if use_embedding:
                    reason_parts.append("high semantic similarity with your description")
                else:
                    reason_parts.append("audio-feature profile and keyword overlap")

            recommendations.append(
                Recommendation(
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    score=final_score,
                    reason="; ".join(reason_parts),
                )
            )

        return recommendations
