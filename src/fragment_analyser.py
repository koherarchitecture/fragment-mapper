"""
Fragment Analyser -- Stage 1 (4 Signals) + Stage 2 (6 Rules)

Four orthogonal signals:
1. Embedding similarity via Hugging Face Inference API (all-MiniLM-L6-v2)
2. TF-IDF distinctiveness via scikit-learn
3. VADER sentiment polarity via nltk
4. Lexical profile via pure Python

Six deterministic rules:
1. Neighbours (clusters) -- agglomerative clustering on embedding similarity
2. Strays (isolates) -- fragments with no semantic neighbours
3. Rifts (tensions) -- same topic, opposing emotional pull
4. Forks (drifts) -- same topic, different emphasis/vocabulary
5. Echoes (resonances) -- different topics, same voice and feeling
6. Shifts (distinctive voices) -- statistically unusual lexical profile

Architecture: AI handles language. Code handles judgment. Humans make decisions.
"""

import asyncio
import logging
import re
from typing import Any

import httpx
import nltk
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

logger = logging.getLogger("fragment-mapper")


# =============================================================================
# Stage 2 Thresholds (all constants, all auditable)
# =============================================================================

# Rule 1: Neighbours
NEIGHBOUR_THRESHOLD = 0.50

# Rule 2: Strays
STRAY_THRESHOLD = 0.20

# Rule 3: Rifts
RIFT_SIM_THRESHOLD = 0.35
RIFT_SENTIMENT_DELTA = 0.40

# Rule 4: Forks
FORK_EMBED_THRESHOLD = 0.40
FORK_TFIDF_THRESHOLD = 0.15

# Rule 5: Echoes
ECHO_EMBED_MAX = 0.25
ECHO_SENTIMENT_WINDOW = 0.15

# Rule 6: Shifts
SHIFT_OUTLIER_THRESHOLD = 1.5


# =============================================================================
# Lexical detection lists
# =============================================================================

HEDGING_WORDS = frozenset({
    "maybe", "perhaps", "might", "could", "possibly", "seems", "appears",
    "somewhat", "relatively", "arguably", "likely", "unlikely", "suggest", "tend",
})

FIRST_PERSON_WORDS = frozenset({
    "i", "me", "my", "we", "our", "mine", "myself", "ourselves",
})

NEGATION_MARKERS = frozenset({
    "not", "never", "without", "no", "don't", "can't", "won't", "isn't", "aren't",
})

OPPOSITION_MARKERS = frozenset({
    "but", "however", "although", "yet", "despite", "contrary", "instead",
})
# "rather than" handled separately as a two-word phrase

ANTONYM_PAIRS = [
    ("control", "freedom"),
    ("structure", "chaos"),
    ("order", "disorder"),
    ("simple", "complex"),
    ("individual", "collective"),
    ("digital", "physical"),
    ("public", "private"),
    ("open", "closed"),
    ("fixed", "fluid"),
    ("visible", "hidden"),
]

# Build a lookup: word -> set of its antonyms
ANTONYM_LOOKUP: dict[str, set[str]] = {}
for a, b in ANTONYM_PAIRS:
    ANTONYM_LOOKUP.setdefault(a, set()).add(b)
    ANTONYM_LOOKUP.setdefault(b, set()).add(a)


# =============================================================================
# HF Inference API URL
# =============================================================================

HF_SIMILARITY_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =============================================================================
# Signal 1: Embedding Similarity
# =============================================================================

async def compute_embeddings(
    fragments: list[str], hf_token: str
) -> tuple[np.ndarray, list[list[float]]]:
    """
    Compute NxN similarity matrix via the HF sentence-similarity API.

    Sends each fragment as source_sentence against all others.
    Returns (similarity_matrix, None) — no raw embeddings available
    from this endpoint.
    """
    n = len(fragments)
    similarity_matrix = np.eye(n, dtype=np.float32)

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(n):
            others = [fragments[j] for j in range(n) if j != i]
            if not others:
                continue

            # Retry up to 3 times — HF cold starts return 503
            last_error = None
            for attempt in range(3):
                response = await client.post(
                    HF_SIMILARITY_URL,
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={
                        "inputs": {
                            "source_sentence": fragments[i],
                            "sentences": others,
                        },
                        "options": {"wait_for_model": True},
                    },
                )
                if response.status_code == 503:
                    last_error = f"HF model loading (attempt {attempt + 1}/3)"
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                response.raise_for_status()
                break
            else:
                raise httpx.HTTPStatusError(
                    last_error or "HF API failed after 3 attempts",
                    request=response.request,
                    response=response,
                )
            scores = response.json()

            # Map scores back to the full matrix (skipping self)
            idx = 0
            for j in range(n):
                if j != i:
                    similarity_matrix[i][j] = float(scores[idx])
                    idx += 1

    return similarity_matrix, None


# =============================================================================
# Signal 2: TF-IDF Distinctiveness
# =============================================================================

def compute_tfidf_signals(
    fragments: list[str],
) -> tuple[list[list[float]], list[list[str]], list[float]]:
    """
    Compute TF-IDF based signals for all fragments.

    Returns:
        (tfidf_similarity, keywords, distinctiveness)
        - tfidf_similarity: NxN cosine similarity matrix (as nested lists)
        - keywords: top 5 distinctive words per fragment
        - distinctiveness: L2 norm of each fragment's TF-IDF vector
    """
    vectoriser = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectoriser.fit_transform(fragments)
    feature_names = vectoriser.get_feature_names_out()

    # Pairwise TF-IDF similarity
    tfidf_sim = sklearn_cosine_similarity(tfidf_matrix).tolist()

    # Top 5 distinctive keywords per fragment
    keywords = []
    for i in range(len(fragments)):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-5:][::-1]
        top_words = [feature_names[j] for j in top_indices if row[j] > 0]
        keywords.append(top_words)

    # Distinctiveness score (L2 norm of TF-IDF vector)
    distinctiveness = [
        float(np.linalg.norm(tfidf_matrix[i].toarray()))
        for i in range(len(fragments))
    ]

    return tfidf_sim, keywords, distinctiveness


# =============================================================================
# Signal 3: VADER Sentiment
# =============================================================================

def compute_sentiment(fragments: list[str]) -> list[dict[str, float]]:
    """
    Compute VADER sentiment for each fragment.

    Returns a list of dicts with keys: compound, positive, negative,
    neutral, intensity.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyser = SentimentIntensityAnalyzer()
    results = []

    for fragment in fragments:
        scores = analyser.polarity_scores(fragment)
        results.append({
            "compound": scores["compound"],
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
            "intensity": abs(scores["compound"]),
        })

    return results


# =============================================================================
# Signal 4: Lexical Profile
# =============================================================================

def compute_lexical_profile(fragment: str) -> dict[str, float]:
    """
    Compute the lexical profile for a single fragment.

    Metrics: question_density, avg_sentence_length, vocabulary_richness,
    first_person_ratio, hedging_ratio, word_count, sentence_count.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", fragment) if s.strip()]
    words = fragment.lower().split()
    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)

    question_marks = fragment.count("?")
    unique_words = len(set(words))

    first_person = sum(1 for w in words if w in FIRST_PERSON_WORDS)
    hedge_count = sum(1 for w in words if w in HEDGING_WORDS)

    return {
        "question_density": question_marks / sentence_count,
        "avg_sentence_length": word_count / sentence_count,
        "vocabulary_richness": unique_words / word_count,
        "first_person_ratio": first_person / word_count,
        "hedging_ratio": hedge_count / sentence_count,
        "sentence_count": sentence_count,
        "word_count": word_count,
    }


def compute_all_lexical_profiles(
    fragments: list[str],
) -> list[dict[str, float]]:
    """Compute lexical profiles for all fragments."""
    return [compute_lexical_profile(f) for f in fragments]


# =============================================================================
# Rule 1: Neighbours (Agglomerative Clustering)
# =============================================================================

def find_neighbours(similarity_matrix: np.ndarray) -> list[list[int]]:
    """
    Find neighbour groups using agglomerative clustering on the
    embedding similarity matrix.

    Uses single-linkage clustering with distance = 1 - similarity.
    Minimum cluster size: 2.
    """
    n = similarity_matrix.shape[0]
    if n < 2:
        return []

    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    # Ensure non-negative distances (floating point can cause tiny negatives)
    distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

    # Extract condensed distance matrix (upper triangle)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(distance_matrix[i, j])
    condensed = np.array(condensed)

    if len(condensed) == 0:
        return []

    # Single-linkage agglomerative clustering
    Z = linkage(condensed, method="single")
    distance_threshold = 1.0 - NEIGHBOUR_THRESHOLD
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Group fragments by cluster label
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    # Only keep clusters with at least 2 members
    return [members for members in clusters.values() if len(members) >= 2]


# =============================================================================
# Rule 2: Strays (Isolates)
# =============================================================================

def find_strays(similarity_matrix: np.ndarray) -> list[int]:
    """
    Find stray fragments -- those whose maximum similarity to any
    other fragment is below STRAY_THRESHOLD.
    """
    n = similarity_matrix.shape[0]
    strays = []

    for i in range(n):
        # Exclude self-similarity
        row = similarity_matrix[i].copy()
        row[i] = -1.0
        max_sim = float(np.max(row))
        if max_sim < STRAY_THRESHOLD:
            strays.append(i)

    return strays


# =============================================================================
# Rule 3: Rifts (Tensions)
# =============================================================================

def _has_negation_opposition(words_a: set[str], words_b: set[str]) -> bool:
    """Check if one fragment has negation markers the other does not."""
    neg_a = words_a & NEGATION_MARKERS
    neg_b = words_b & NEGATION_MARKERS
    # One has negation, the other does not
    return bool(neg_a) != bool(neg_b)


def _has_opposition_markers(words_a: set[str], words_b: set[str]) -> bool:
    """Check if either fragment contains opposition markers."""
    all_words = words_a | words_b
    return bool(all_words & OPPOSITION_MARKERS) or (
        "rather" in all_words and "than" in all_words
    )


def _find_antonym_pairs(
    words_a: set[str], words_b: set[str]
) -> list[str]:
    """Find antonym pairs where one word is in fragment A and its antonym in B."""
    found = []
    for word_a in words_a:
        if word_a in ANTONYM_LOOKUP:
            matching = ANTONYM_LOOKUP[word_a] & words_b
            for word_b in matching:
                # Canonical order (alphabetical)
                pair = "/".join(sorted([word_a, word_b]))
                if pair not in found:
                    found.append(pair)
    return found


def find_rifts(
    similarity_matrix: np.ndarray,
    sentiment: list[dict[str, float]],
    fragments: list[str],
) -> list[dict[str, Any]]:
    """
    Find rift pairs: same topic (high embedding similarity) but
    opposing emotional pull (sentiment opposition or lexical opposition).
    """
    n = len(fragments)
    rifts = []

    # Pre-compute word sets for lexical opposition checks
    word_sets = [set(f.lower().split()) for f in fragments]

    for i in range(n):
        for j in range(i + 1, n):
            embed_sim = float(similarity_matrix[i, j])

            # Must be topically related
            if embed_sim < RIFT_SIM_THRESHOLD:
                continue

            signals = []

            # Check sentiment opposition
            compound_a = sentiment[i]["compound"]
            compound_b = sentiment[j]["compound"]
            sentiment_delta = abs(compound_a - compound_b)

            has_sentiment_opposition = (
                sentiment_delta >= RIFT_SENTIMENT_DELTA
                and (
                    (compound_a > 0.1 and compound_b < -0.1)
                    or (compound_a < -0.1 and compound_b > 0.1)
                )
            )

            if has_sentiment_opposition:
                signals.append("sentiment_opposition")

            # Check lexical opposition
            words_a = word_sets[i]
            words_b = word_sets[j]

            if _has_negation_opposition(words_a, words_b):
                signals.append("negation_opposition")

            if _has_opposition_markers(words_a, words_b):
                signals.append("opposition_markers")

            antonym_pairs = _find_antonym_pairs(words_a, words_b)
            for pair in antonym_pairs:
                signals.append(pair)

            has_lexical_opposition = len(signals) > (
                1 if has_sentiment_opposition else 0
            )

            # A rift requires topical similarity AND (sentiment OR lexical opposition)
            if has_sentiment_opposition or has_lexical_opposition:
                rifts.append({
                    "pair": [i, j],
                    "embedding_sim": round(embed_sim, 4),
                    "sentiment_delta": round(sentiment_delta, 4),
                    "signals": signals,
                })

    return rifts


# =============================================================================
# Rule 4: Forks (Drifts)
# =============================================================================

def find_forks(
    similarity_matrix: np.ndarray,
    tfidf_similarity: list[list[float]],
) -> list[dict[str, Any]]:
    """
    Find fork pairs: high embedding similarity (same topic) but low
    TF-IDF similarity (different emphasis/vocabulary).
    """
    n = similarity_matrix.shape[0]
    forks = []

    for i in range(n):
        for j in range(i + 1, n):
            embed_sim = float(similarity_matrix[i, j])
            tfidf_sim = tfidf_similarity[i][j]

            if embed_sim >= FORK_EMBED_THRESHOLD and tfidf_sim <= FORK_TFIDF_THRESHOLD:
                forks.append({
                    "pair": [i, j],
                    "embedding_sim": round(embed_sim, 4),
                    "tfidf_sim": round(tfidf_sim, 4),
                    "fork_magnitude": round(embed_sim - tfidf_sim, 4),
                })

    return forks


# =============================================================================
# Rule 5: Echoes (Resonances)
# =============================================================================

def _lexical_trait_name(metric: str) -> str:
    """Map lexical metric name to a human-readable trait name."""
    mapping = {
        "question_density": "questioning",
        "hedging_ratio": "hedging",
        "vocabulary_richness": "vocabulary_richness",
    }
    return mapping.get(metric, metric)


def find_echoes(
    similarity_matrix: np.ndarray,
    sentiment: list[dict[str, float]],
    lexical_profiles: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """
    Find echo pairs: low embedding similarity (different topics) but
    similar sentiment and at least one shared lexical trait.
    """
    n = similarity_matrix.shape[0]
    echoes = []

    # Traits to compare for lexical alignment
    trait_metrics = ["question_density", "hedging_ratio", "vocabulary_richness"]

    for i in range(n):
        for j in range(i + 1, n):
            embed_sim = float(similarity_matrix[i, j])

            # Must NOT be topically similar
            if embed_sim >= ECHO_EMBED_MAX:
                continue

            # Check sentiment proximity
            compound_diff = abs(
                sentiment[i]["compound"] - sentiment[j]["compound"]
            )
            if compound_diff > ECHO_SENTIMENT_WINDOW:
                continue

            # Check shared lexical traits
            shared_traits = []
            for metric in trait_metrics:
                val_a = lexical_profiles[i].get(metric, 0.0)
                val_b = lexical_profiles[j].get(metric, 0.0)
                if abs(val_a - val_b) <= 0.15:
                    shared_traits.append(_lexical_trait_name(metric))

            if not shared_traits:
                continue

            sentiment_proximity = 1.0 - compound_diff
            echoes.append({
                "pair": [i, j],
                "embedding_sim": round(embed_sim, 4),
                "sentiment_proximity": round(sentiment_proximity, 4),
                "shared_traits": shared_traits,
            })

    return echoes


# =============================================================================
# Rule 6: Shifts (Distinctive Voices)
# =============================================================================

def find_shifts(
    lexical_profiles: list[dict[str, float]],
) -> list[int]:
    """
    Find shift fragments: those whose lexical profile is statistically
    unusual (> SHIFT_OUTLIER_THRESHOLD std devs from centroid).
    """
    if len(lexical_profiles) < 3:
        # Need enough fragments to compute meaningful statistics
        return []

    # Use the profile metrics that characterise voice
    profile_keys = [
        "question_density",
        "avg_sentence_length",
        "vocabulary_richness",
        "first_person_ratio",
        "hedging_ratio",
    ]

    # Build matrix of profile values
    matrix = np.array([
        [profile.get(k, 0.0) for k in profile_keys]
        for profile in lexical_profiles
    ], dtype=np.float64)

    # Normalise each dimension to [0, 1] range to prevent sentence length
    # from dominating the distance calculation
    col_min = matrix.min(axis=0)
    col_max = matrix.max(axis=0)
    col_range = col_max - col_min
    # Avoid division by zero for constant columns
    col_range = np.where(col_range == 0, 1.0, col_range)
    normalised = (matrix - col_min) / col_range

    # Compute centroid
    centroid = normalised.mean(axis=0)

    # Compute Euclidean distance from centroid for each fragment
    distances = np.linalg.norm(normalised - centroid, axis=1)

    # Flag outliers
    mean_distance = distances.mean()
    std_distance = distances.std()

    if std_distance == 0:
        return []

    threshold = mean_distance + SHIFT_OUTLIER_THRESHOLD * std_distance
    shifts = [int(i) for i in range(len(distances)) if distances[i] > threshold]

    return shifts


# =============================================================================
# Unclustered: fragments not in any neighbour group and not strays
# =============================================================================

def find_unclustered(
    n: int,
    neighbours: list[list[int]],
    strays: list[int],
) -> list[int]:
    """
    Find fragments that are neither in a neighbour group nor strays.
    These have moderate connections but no strong group membership.
    """
    clustered = set()
    for group in neighbours:
        clustered.update(group)

    stray_set = set(strays)
    all_indices = set(range(n))

    return sorted(all_indices - clustered - stray_set)


# =============================================================================
# FragmentAnalyser: The main class
# =============================================================================

class FragmentAnalyser:
    """
    Runs the four-signal analysis (Stage 1) and six-rule evaluation
    (Stage 2) on a set of text fragments.

    Embedding is performed via the Hugging Face Inference API (async).
    All other signals are computed locally.
    """

    def __init__(self, hf_token: str):
        """
        Initialise the analyser.

        Args:
            hf_token: Hugging Face API token for embedding inference.
        """
        self.hf_token = hf_token

        # Ensure VADER lexicon is available
        nltk.download("vader_lexicon", quiet=True)

        logger.info("FragmentAnalyser initialised (VADER lexicon ready)")

    async def analyse(self, fragments: list[str]) -> dict[str, Any]:
        """
        Run the full analysis pipeline on the given fragments.

        Stage 1: Compute four orthogonal signals.
        Stage 2: Apply six deterministic rules.

        Args:
            fragments: List of text fragments (2-20 items, 5-200 words each).

        Returns:
            Complete analysis dict with signals and rule results.
        """
        n = len(fragments)
        logger.info(f"Analysing {n} fragments")

        # =====================================================================
        # Stage 1: Four Signals
        # =====================================================================

        # Signal 1: Embedding similarity (async -- HF API call)
        logger.info("Signal 1: Computing embedding similarity via HF API")
        similarity_matrix, _embeddings = await compute_embeddings(
            fragments, self.hf_token
        )

        # Signal 2: TF-IDF distinctiveness (sync -- scikit-learn)
        logger.info("Signal 2: Computing TF-IDF distinctiveness")
        tfidf_similarity, keywords, distinctiveness = compute_tfidf_signals(
            fragments
        )

        # Signal 3: VADER sentiment (sync -- nltk lexicon)
        logger.info("Signal 3: Computing VADER sentiment")
        sentiment = compute_sentiment(fragments)

        # Signal 4: Lexical profiles (sync -- pure Python)
        logger.info("Signal 4: Computing lexical profiles")
        lexical_profiles = compute_all_lexical_profiles(fragments)

        # =====================================================================
        # Stage 2: Six Rules
        # =====================================================================

        # Rule 1: Neighbours
        logger.info("Rule 1: Finding neighbours")
        neighbours = find_neighbours(similarity_matrix)

        # Rule 2: Strays
        logger.info("Rule 2: Finding strays")
        strays = find_strays(similarity_matrix)

        # Rule 3: Rifts
        logger.info("Rule 3: Finding rifts")
        rifts = find_rifts(similarity_matrix, sentiment, fragments)

        # Rule 4: Forks
        logger.info("Rule 4: Finding forks")
        forks = find_forks(similarity_matrix, tfidf_similarity)

        # Rule 5: Echoes
        logger.info("Rule 5: Finding echoes")
        echoes = find_echoes(similarity_matrix, sentiment, lexical_profiles)

        # Rule 6: Shifts
        logger.info("Rule 6: Finding shifts")
        shifts = find_shifts(lexical_profiles)

        # Unclustered
        unclustered = find_unclustered(n, neighbours, strays)

        logger.info(
            f"Analysis complete: {len(neighbours)} neighbour groups, "
            f"{len(strays)} strays, {len(rifts)} rifts, {len(forks)} forks, "
            f"{len(echoes)} echoes, {len(shifts)} shifts"
        )

        # =====================================================================
        # Assemble result
        # =====================================================================

        return {
            "fragment_count": n,
            "fragments": fragments,
            # Stage 2 rule results
            "neighbours": neighbours,
            "strays": strays,
            "rifts": rifts,
            "forks": forks,
            "echoes": echoes,
            "shifts": shifts,
            "unclustered": unclustered,
            # Stage 1 signal data
            "similarity_matrix": similarity_matrix.tolist(),
            "tfidf_similarity": tfidf_similarity,
            "sentiment": sentiment,
            "lexical_profiles": lexical_profiles,
            "keywords": keywords,
            "distinctiveness": distinctiveness,
        }
