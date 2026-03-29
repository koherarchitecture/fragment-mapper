#!/usr/bin/env python3
"""
Fragment Mapper — Study Module

A step-by-step walkthrough of how the tool works.
Designed for design students, not computer scientists.

Run this to see the system in action:
    python study.py

Or explore specific stages:
    python study.py --stage 1
    python study.py --stage 2
    python study.py --stage 3
    python study.py --all

No API keys required. All examples run locally.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.fragment_analyser import (
    compute_tfidf_signals,
    compute_sentiment,
    compute_all_lexical_profiles,
    find_neighbours,
    find_strays,
    find_rifts,
    find_forks,
    find_echoes,
    find_shifts,
    find_unclustered,
    NEIGHBOUR_THRESHOLD,
    STRAY_THRESHOLD,
    RIFT_SIM_THRESHOLD,
    RIFT_SENTIMENT_DELTA,
    FORK_EMBED_THRESHOLD,
    FORK_TFIDF_THRESHOLD,
    ECHO_EMBED_MAX,
    ECHO_SENTIMENT_WINDOW,
    SHIFT_OUTLIER_THRESHOLD,
)

import numpy as np


# =============================================================================
# EXAMPLE FRAGMENTS
# =============================================================================

EXAMPLE_FRAGMENTS = [
    "Students learn more from confusion than from clarity. The moment something doesn't make sense is when the real thinking starts.",
    "A good lecture should be so well structured that every student follows every step. Confusion means the teacher failed to prepare.",
    "The best feedback I ever received was a single question that I couldn't answer for three weeks.",
    "Assessment rubrics make grading fair but they also flatten the work. Everything becomes a checklist.",
    "When I watch students discuss each other's projects, they say things no teacher would think to say. The vocabulary is different.",
]

# Pre-computed similarity matrix (from all-MiniLM-L6-v2 embeddings)
# This avoids requiring an HF token for study mode.
EXAMPLE_SIMILARITY = np.array([
    [1.0000, 0.5836, 0.3294, 0.2705, 0.4553],
    [0.5836, 1.0000, 0.2109, 0.2826, 0.3421],
    [0.3294, 0.2109, 1.0000, 0.1587, 0.2043],
    [0.2705, 0.2826, 0.1587, 1.0000, 0.3198],
    [0.4553, 0.3421, 0.2043, 0.3198, 1.0000],
], dtype=np.float32)


# =============================================================================
# VISUAL HELPERS
# =============================================================================

def clear_screen():
    """Clear terminal for fresh start."""
    print("\033[2J\033[H", end="")


def pause(prompt="Press Enter to continue..."):
    """Wait for user input."""
    input(f"\n{prompt}")


def print_header(title: str):
    """Print a section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


def print_subheader(title: str):
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")
    print()


def print_box(content: str, title: str = ""):
    """Print content in a box."""
    lines = content.strip().split("\n")
    width = max(len(line) for line in lines) + 4
    width = max(width, len(title) + 6)

    print()
    if title:
        print(f"+-- {title} " + "-" * (width - len(title) - 5) + "+")
    else:
        print("+" + "-" * width + "+")

    for line in lines:
        padding = width - len(line) - 2
        print(f"| {line}" + " " * max(padding, 0) + "|")

    print("+" + "-" * width + "+")
    print()


def print_arrow():
    """Print a downward arrow to show flow."""
    print("            |")
    print("            v")


def print_fragment(idx: int, text: str, max_width: int = 55):
    """Print a fragment with wrapping."""
    words = text.split()
    lines = []
    line = ""
    for w in words:
        if len(line) + len(w) + 1 > max_width and line:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)

    print(f"  [{idx + 1}] {lines[0]}")
    for l in lines[1:]:
        print(f"      {l}")


# =============================================================================
# INTRODUCTION
# =============================================================================

def show_introduction():
    """Explain what this tool is and why it exists."""
    clear_screen()
    print_header("FRAGMENT MAPPER -- STUDY MODULE")

    print("""This tool shows the structural relationships between
scattered text fragments.

Not what you should write. Not what's good or bad.
The STRUCTURE of how your fragments relate to each other.

Where do they cluster? Where do they pull apart?
Where does the same voice appear across different topics?
""")

    print_box("""The core insight:

When you write fragments -- scattered thoughts, notes,
half-formed ideas -- they have STRUCTURE you can't see.

Some fragments are neighbours (same topic).
Some are strays (isolated, no connections).
Some are rifts (same topic, opposing feelings).
Some echo each other across different topics.

This tool makes that structure visible.""", title="Why?")

    pause()

    print_subheader("THE THREE STAGES")

    print("""This tool works in three stages:

  STAGE 1: QUALIFICATION (four signals)
  AI reads your fragments and extracts four measurements:
  embedding similarity, TF-IDF distinctiveness,
  sentiment polarity, and lexical profile.

  STAGE 2: RULES (six rules)
  Deterministic code applies thresholds to the signals.
  Same input ALWAYS gives same output. No AI here.

  STAGE 3: LANGUAGE (narration)
  An AI (Claude Haiku) describes what the rules found.
  It observes -- it doesn't judge. The analysis was
  already done in Stage 2.
""")

    print_box("""The principle:

    AI handles LANGUAGE.
    Code handles JUDGMENT.
    Humans make DECISIONS.

The AI never decides if your fragments are 'good'.
It only describes the relationships that emerge.""", title="Architecture")

    pause()


# =============================================================================
# STAGE 1: THE FOUR SIGNALS
# =============================================================================

def show_stage_1():
    """Explain and demonstrate the four signals."""
    clear_screen()
    print_header("STAGE 1: FOUR SIGNALS")

    print("""Stage 1 reads your fragments and extracts four
independent measurements. Each captures something different.

We'll use these five fragments as our example:
""")

    for i, f in enumerate(EXAMPLE_FRAGMENTS):
        print_fragment(i, f)
        print()

    pause()

    # Signal 1: Embedding similarity
    print_subheader("SIGNAL 1: EMBEDDING SIMILARITY")

    print("""Each fragment is converted into a vector (a list of
numbers) that captures its MEANING. Fragments about
similar topics have similar vectors.

The similarity score runs from 0 (completely unrelated)
to 1 (identical meaning).

Here is the similarity matrix for our five fragments:
""")

    # Print matrix
    print("       ", end="")
    for j in range(5):
        print(f"  [{j+1}]  ", end="")
    print()

    for i in range(5):
        print(f"  [{i+1}] ", end="")
        for j in range(5):
            val = EXAMPLE_SIMILARITY[i, j]
            if i == j:
                print("  ---  ", end="")
            elif val >= 0.5:
                print(f" {val:.2f}* ", end="")
            else:
                print(f" {val:.2f}  ", end="")
        print()

    print("\n  (* = high similarity)")

    print("""
Notice: Fragments 1 and 2 have the highest similarity (0.58).
They're both about confusion and learning. But they say
OPPOSITE things about it.

The embedding captures topic. Not opinion.""")

    pause()

    # Signal 2: TF-IDF
    print_subheader("SIGNAL 2: TF-IDF DISTINCTIVENESS")

    tfidf_sim, keywords, distinctiveness = compute_tfidf_signals(
        EXAMPLE_FRAGMENTS
    )

    print("""TF-IDF measures what makes each fragment UNIQUE
within this particular set. It finds the words that
distinguish one fragment from the others.
""")

    for i in range(5):
        kw = ", ".join(keywords[i][:4])
        print(f"  [{i+1}] Keywords: {kw}")
        print(f"      Distinctiveness: {distinctiveness[i]:.3f}")
        print()

    print("""Higher distinctiveness = more unique vocabulary.
The keywords show what sets each fragment apart.""")

    pause()

    # Signal 3: Sentiment
    print_subheader("SIGNAL 3: SENTIMENT POLARITY (VADER)")

    sentiment = compute_sentiment(EXAMPLE_FRAGMENTS)

    print("""VADER reads emotional temperature. The compound score
runs from -1 (strongly negative) to +1 (strongly positive).
0 is neutral.
""")

    for i in range(5):
        s = sentiment[i]
        compound = s["compound"]
        bar_pos = int((compound + 1) / 2 * 20)
        bar = "-" * bar_pos + "|" + "-" * (20 - bar_pos)

        label = "positive" if compound > 0.05 else "negative" if compound < -0.05 else "neutral"
        print(f"  [{i+1}] [{bar}] {compound:+.3f} ({label})")

    print("""
Fragment 1 is positive (confusion = real thinking).
Fragment 2 is negative (confusion = teacher failure).
Same topic. Opposite feelings. That's a RIFT.""")

    pause()

    # Signal 4: Lexical profile
    print_subheader("SIGNAL 4: LEXICAL PROFILE")

    profiles = compute_all_lexical_profiles(EXAMPLE_FRAGMENTS)

    print("""The lexical profile captures WRITING CHARACTER --
not what you say, but how you say it.
""")

    for i in range(5):
        p = profiles[i]
        traits = []
        if p["question_density"] > 0.2:
            traits.append("questioning")
        if p["hedging_ratio"] > 0.2:
            traits.append("hedged")
        if p["first_person_ratio"] > 0.08:
            traits.append("personal")
        if p["avg_sentence_length"] > 20:
            traits.append("long sentences")
        elif p["avg_sentence_length"] < 10:
            traits.append("short sentences")
        if p["vocabulary_richness"] > 0.85:
            traits.append("diverse vocabulary")

        trait_str = ", ".join(traits) if traits else "neutral"
        print(f"  [{i+1}] Voice: {trait_str}")
        print(f"      Sentence length: {p['avg_sentence_length']:.0f} words")
        print(f"      First person: {p['first_person_ratio']:.0%}")
        print()

    print("""Fragment 3 is personal ("I ever received", "I couldn't").
The others are declarative. That's a different voice --
the rules will flag it as a SHIFT.""")

    pause()


# =============================================================================
# STAGE 2: THE SIX RULES
# =============================================================================

def show_stage_2():
    """Explain and demonstrate the six rules."""
    clear_screen()
    print_header("STAGE 2: SIX DETERMINISTIC RULES")

    print("""Stage 2 takes the four signals and applies six rules.
Each rule has explicit thresholds -- constants in the
source code that you can read and change.

No AI here. Same input ALWAYS gives same output.
""")

    # Compute all signals
    sentiment = compute_sentiment(EXAMPLE_FRAGMENTS)
    tfidf_sim, keywords, distinctiveness = compute_tfidf_signals(
        EXAMPLE_FRAGMENTS
    )
    profiles = compute_all_lexical_profiles(EXAMPLE_FRAGMENTS)
    sim = EXAMPLE_SIMILARITY
    n = len(EXAMPLE_FRAGMENTS)

    pause()

    # Rule 1: Neighbours
    print_subheader("RULE 1: NEIGHBOURS")

    print(f"""Threshold: similarity >= {NEIGHBOUR_THRESHOLD}

Neighbours are groups of fragments that share a topic.
Found by agglomerative clustering on the similarity matrix.
""")

    neighbours = find_neighbours(sim)
    if neighbours:
        for group in neighbours:
            members = ", ".join(f"[{i+1}]" for i in group)
            print(f"  Group: {members}")
    else:
        print("  No neighbour groups found.")

    pause()

    # Rule 2: Strays
    print_subheader("RULE 2: STRAYS")

    print(f"""Threshold: max similarity to any other < {STRAY_THRESHOLD}

Strays are fragments with no close neighbours. They stand
alone -- not necessarily a problem, just a fact.
""")

    strays = find_strays(sim)
    if strays:
        for i in strays:
            max_sim = max(sim[i, j] for j in range(n) if j != i)
            print(f"  [{i+1}] (max similarity: {max_sim:.3f})")
    else:
        print("  No strays found.")

    pause()

    # Rule 3: Rifts
    print_subheader("RULE 3: RIFTS")

    print(f"""Thresholds: embedding >= {RIFT_SIM_THRESHOLD} AND
            sentiment delta >= {RIFT_SENTIMENT_DELTA}

Rifts are fragments about the SAME topic that pull in
OPPOSITE emotional directions. Tension worth examining.
""")

    rifts = find_rifts(sim, sentiment, EXAMPLE_FRAGMENTS)
    if rifts:
        for r in rifts:
            i, j = r["pair"]
            print(f"  [{i+1}] vs [{j+1}]")
            print(f"    Embedding similarity: {r['embedding_sim']:.3f}")
            print(f"    Sentiment delta: {r['sentiment_delta']:.3f}")
            print(f"    Signals: {', '.join(r['signals'])}")
    else:
        print("  No rifts found.")

    pause()

    # Rule 4: Forks
    print_subheader("RULE 4: FORKS")

    print(f"""Thresholds: embedding >= {FORK_EMBED_THRESHOLD} AND
            TF-IDF similarity <= {FORK_TFIDF_THRESHOLD}

Forks are fragments about the SAME topic but using
DIFFERENT vocabulary. Potential for synthesis.
""")

    forks = find_forks(sim, tfidf_sim)
    if forks:
        for f in forks:
            i, j = f["pair"]
            print(f"  [{i+1}] vs [{j+1}]")
            print(f"    Embedding: {f['embedding_sim']:.3f} (topic match)")
            print(f"    TF-IDF: {f['tfidf_sim']:.3f} (vocabulary gap)")
    else:
        print("  No forks found.")

    pause()

    # Rule 5: Echoes
    print_subheader("RULE 5: ECHOES")

    print(f"""Thresholds: embedding < {ECHO_EMBED_MAX} AND
            sentiment window <= {ECHO_SENTIMENT_WINDOW}
            AND shared lexical traits

Echoes are fragments about DIFFERENT topics that share
the same VOICE. An unconscious pattern.
""")

    echoes = find_echoes(sim, sentiment, profiles)
    if echoes:
        for e in echoes:
            i, j = e["pair"]
            print(f"  [{i+1}] vs [{j+1}]")
            print(f"    Embedding: {e['embedding_sim']:.3f} (different topics)")
            print(f"    Shared traits: {', '.join(e['shared_traits'])}")
    else:
        print("  No echoes found.")

    pause()

    # Rule 6: Shifts
    print_subheader("RULE 6: SHIFTS")

    print(f"""Threshold: lexical distance > {SHIFT_OUTLIER_THRESHOLD} std devs
            from group centroid

Shifts are fragments whose WRITING CHARACTER is markedly
different from the others. A change in voice.
""")

    shifts = find_shifts(profiles)
    if shifts:
        for i in shifts:
            p = profiles[i]
            print(f"  [{i+1}] (outlier voice)")
            print(f"    First person: {p['first_person_ratio']:.0%}")
            print(f"    Sentence length: {p['avg_sentence_length']:.0f}")
    else:
        print("  No shifts found.")

    pause()

    # Summary
    print_subheader("SUMMARY")

    unclustered = find_unclustered(n, neighbours, strays)

    print(f"""  Neighbours: {len(neighbours)} group(s)
  Strays:     {len(strays)}
  Rifts:      {len(rifts)}
  Forks:      {len(forks)}
  Echoes:     {len(echoes)}
  Shifts:     {len(shifts)}

Every one of these findings can be traced to:
  - A signal value (from Stage 1)
  - A threshold (a constant in the source code)
  - A rule (a function you can read)

No black box. No 'the AI decided'.""")

    pause()


# =============================================================================
# STAGE 3: THE NARRATOR
# =============================================================================

def show_stage_3():
    """Explain how the AI narrator works."""
    clear_screen()
    print_header("STAGE 3: LANGUAGE GENERATION")

    print("""Stage 3 is where AI re-enters -- but only for LANGUAGE.

The analysis is done. The rules have fired.
Now we need to DESCRIBE what was found in plain language.
""")

    print_subheader("WHAT THE NARRATOR RECEIVES")

    print("""The narrator (Claude Haiku) receives:

  1. The numbered fragments (your text)
  2. Which fragments are neighbours
  3. Which are strays
  4. Any rifts, forks, echoes, shifts found
  5. The similarity scores and sentiment data

It does NOT receive:
  - Instructions to judge quality
  - Other users' fragments
  - Any ability to change the rule findings
""")

    pause()

    print_subheader("THE SYSTEM PROMPT")

    print_box("""You describe the structural relationships between
text fragments that a rules engine has identified.
You are observational, like someone describing a map.
You are a mirror, not a mentor.

Use the author's own words -- quote brief phrases.
Do not evaluate, prescribe, or suggest.
Do not introduce findings the rules did not produce.""",
    title="What the AI is told")

    pause()

    print_subheader("WHY LANGUAGE COMES LAST")

    print("""The sequence matters:

    STAGE 1: AI reads language (signals)
         |
    STAGE 2: Code makes judgments (rules)
         |
    STAGE 3: AI describes what emerged (narration)

If we let AI do the judging:
  - Results would vary unpredictably
  - We couldn't explain WHY something was flagged
  - Users would have to trust a black box

By separating the layers:
  - Stage 2 is AUDITABLE (read the thresholds)
  - Stage 3 is CONSTRAINED (can only describe)
  - The system is TRUSTWORTHY (no hidden judgments)
""")

    print("""To see Stage 3 in action, run the full tool:

    cp config.env.example config.env
    # Add your HF_TOKEN and OPENROUTER_API_KEY
    PYTHONPATH=. uvicorn backend.main:app
    # Open http://localhost:8000
""")

    pause()


# =============================================================================
# INTERACTIVE EXPLORER
# =============================================================================

def interactive_explorer():
    """Let users try their own fragments."""
    clear_screen()
    print_header("INTERACTIVE EXPLORER")

    print("""Enter your own fragments to see the rules in action.

Type each fragment on its own line.
Enter a blank line when done (minimum 3 fragments).
Type 'q' to quit.
""")

    while True:
        print("\nEnter fragments (blank line to analyse, 'q' to quit):\n")
        fragments = []
        while True:
            line = input("  > ").strip()
            if line.lower() == 'q':
                return
            if not line:
                break
            if len(line.split()) >= 5:
                fragments.append(line)
                print(f"    (fragment {len(fragments)} added)")
            else:
                print("    (too short -- need at least 5 words)")

        if len(fragments) < 3:
            print("\n  Need at least 3 fragments.")
            continue

        print(f"\n  Analysing {len(fragments)} fragments...\n")

        # Compute signals (no HF API -- skip embedding)
        sentiment = compute_sentiment(fragments)
        tfidf_sim, keywords, distinctiveness = compute_tfidf_signals(fragments)
        profiles = compute_all_lexical_profiles(fragments)

        # For rules that need embeddings, use TF-IDF similarity as proxy
        n = len(fragments)
        sim_proxy = np.array(tfidf_sim, dtype=np.float32)

        print("  SIGNALS:")
        for i in range(n):
            s = sentiment[i]
            kw = ", ".join(keywords[i][:3])
            print(f"    [{i+1}] sentiment: {s['compound']:+.2f} | keywords: {kw}")
        print()

        # Apply rules with proxy similarity
        neighbours = find_neighbours(sim_proxy)
        strays = find_strays(sim_proxy)
        rifts = find_rifts(sim_proxy, sentiment, fragments)
        forks = find_forks(sim_proxy, tfidf_sim)
        echoes = find_echoes(sim_proxy, sentiment, profiles)
        shifts = find_shifts(profiles)

        print("  RULES FIRED:")
        print(f"    Neighbours: {len(neighbours)}")
        print(f"    Strays:     {len(strays)}")
        print(f"    Rifts:      {len(rifts)}")
        print(f"    Forks:      {len(forks)}")
        print(f"    Echoes:     {len(echoes)}")
        print(f"    Shifts:     {len(shifts)}")

        if neighbours:
            for g in neighbours:
                print(f"    -> Neighbour group: {[i+1 for i in g]}")
        if strays:
            print(f"    -> Strays: {[i+1 for i in strays]}")
        if rifts:
            for r in rifts:
                print(f"    -> Rift: [{r['pair'][0]+1}] vs [{r['pair'][1]+1}]")
        if shifts:
            print(f"    -> Shifts: {[i+1 for i in shifts]}")

        print("\n  (Note: explorer uses TF-IDF as similarity proxy.")
        print("   Full tool uses HF embeddings for more accurate results.)")


# =============================================================================
# MAIN
# =============================================================================

def show_help():
    """Show usage information."""
    print("""
Fragment Mapper -- Study Module

Usage:
    python study.py              Full walkthrough (recommended first time)
    python study.py --all        Same as above
    python study.py --stage 1    Stage 1: The four signals
    python study.py --stage 2    Stage 2: The six rules
    python study.py --stage 3    Stage 3: The narrator
    python study.py --explore    Interactive fragment explorer
    python study.py --help       Show this message

This module is designed for DESIGN STUDENTS -- it explains
the WHY, not just the WHAT.

No API keys required. All examples run locally.
""")


def main():
    """Main entry point."""
    # Ensure VADER lexicon is available
    import nltk
    nltk.download("vader_lexicon", quiet=True)

    args = sys.argv[1:]

    if not args or "--all" in args:
        show_introduction()
        show_stage_1()
        show_stage_2()
        show_stage_3()

        print_header("WHAT NEXT?")
        print("""You've seen how the system works.

To explore more:
    python study.py --explore    Try your own fragments
    python study.py --stage 2    Deep dive on rules

To use the actual tool:
    cp config.env.example config.env
    # Add your HF_TOKEN and OPENROUTER_API_KEY
    PYTHONPATH=. uvicorn backend.main:app
    Then open http://localhost:8000

To experiment with thresholds:
    Open src/fragment_analyser.py
    Change a threshold constant
    Run study.py again -- see what changes
""")

    elif "--help" in args or "-h" in args:
        show_help()

    elif "--stage" in args:
        try:
            idx = args.index("--stage")
            stage = int(args[idx + 1])
        except (ValueError, IndexError):
            print("Usage: python study.py --stage [1|2|3]")
            return

        if stage == 1:
            show_stage_1()
        elif stage == 2:
            show_stage_2()
        elif stage == 3:
            show_stage_3()
        else:
            print("Stage must be 1, 2, or 3")

    elif "--explore" in args:
        interactive_explorer()

    else:
        show_help()


if __name__ == "__main__":
    main()
