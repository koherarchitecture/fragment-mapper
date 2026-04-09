"""
Stage 3: Haiku narrator for Fragment Mapper v3.

Describes the structural relationships between fragments using findings
from the six deterministic rules. Does not evaluate, prescribe, or
generate content. Observational, not prescriptive.

Uses Claude Haiku 4.5 via OpenRouter.
"""

import logging
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI

logger = logging.getLogger("fragment-mapper")


SYSTEM_PROMPT = """You describe the structural relationships between text fragments that a rules engine has identified. You are observational, like someone describing a map. You are a mirror, not a mentor.

Use the author's own words -- quote brief phrases from their fragments, do not paraphrase.

For neighbours (clusters): describe what the fragments share. Name the territory they occupy using their words.
For unclustered fragments: these have moderate connections -- related enough to not be isolated, but not similar enough to cluster. Note the in-between quality without treating it as a problem.
For strays (isolates): note their distance without judging. A stray fragment is not a problem -- it is a fact.
For rifts (tensions): describe the pull in both directions. Name the emotional opposition. Do not resolve the tension.
For forks: describe the shared topic and the different emphasis. Name the vocabulary gap.
For echoes (resonances): describe the shared voice across different topics. Name what the fragments share that is not topic.
For shifts (distinctive voices): name what changes. "This fragment asks where the others declare."

Important: when there are no neighbour clusters but fragments appear topically related (unclustered), do not say "no clusters" without qualification. The visual map positions fragments by raw similarity, so related-but-unclustered fragments will appear close together. Instead, describe the relationship accurately: fragments are related but below the clustering threshold -- they share territory without forming a defined group.

Do not evaluate, prescribe, or suggest. Do not introduce findings the rules did not produce. Tone: observational. Length: 2-4 sentences for 3-5 fragments, up to 10 for 15-20 fragments.

After describing the map, end with 1-3 questions that arise from the specific relationships found. These are genuine questions, not suggestions disguised as questions. Frame them as "what the map makes you wonder" -- e.g. "The rift between fragments 2 and 4 pulls in opposite directions on the same ground — is that a contradiction you want to resolve, or a tension you want to hold?" or "Fragments 1 and 3 use different words for what the embeddings see as the same territory — which version would you say to someone who doesn't know this project?" Never ask generic questions. Every question must reference specific fragments and specific findings."""


class FragmentNarrator:
    """Generates structural descriptions using Claude Haiku via OpenRouter."""

    def __init__(self, api_key: str | None = None):
        """
        Initialise with OpenRouter API key.

        Args:
            api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for narration")

        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        self.model = "anthropic/claude-haiku-4.5"

    def _build_user_prompt(self, analysis: dict) -> str:
        """
        Build the user prompt from the analysis output.

        Includes the numbered fragments and all Stage 2 findings
        with their evidence.
        """
        fragments = analysis["fragments"]

        # -- Fragments (numbered) --
        fragment_lines = []
        for i, f in enumerate(fragments):
            fragment_lines.append(f'Fragment {i + 1}: "{f}"')

        sections = [
            "Here are the fragments and the structural relationships the rules found:",
            "",
            "**Fragments:**",
            "\n".join(fragment_lines),
        ]

        # -- Neighbours --
        neighbours = analysis.get("neighbours", [])
        if neighbours:
            lines = []
            for group in neighbours:
                members = ", ".join(f"Fragment {idx + 1}" for idx in group)
                lines.append(f"- Group: {members}")
            sections.append("")
            sections.append("**Neighbours (semantic clusters):**")
            sections.append("\n".join(lines))

        # -- Strays --
        strays = analysis.get("strays", [])
        if strays:
            stray_text = ", ".join(f"Fragment {idx + 1}" for idx in strays)
            sections.append("")
            sections.append(f"**Strays (isolates):** {stray_text}")

        # -- Rifts --
        rifts = analysis.get("rifts", [])
        if rifts:
            lines = []
            for rift in rifts:
                pair = rift["pair"]
                signals = ", ".join(rift.get("signals", []))
                lines.append(
                    f"- Fragment {pair[0] + 1} and Fragment {pair[1] + 1} "
                    f"(embedding similarity: {rift['embedding_sim']:.2f}, "
                    f"sentiment delta: {rift['sentiment_delta']:.2f}, "
                    f"signals: {signals})"
                )
            sections.append("")
            sections.append("**Rifts (tensions -- same topic, opposing pull):**")
            sections.append("\n".join(lines))

        # -- Forks --
        forks = analysis.get("forks", [])
        if forks:
            lines = []
            for fork in forks:
                pair = fork["pair"]
                lines.append(
                    f"- Fragment {pair[0] + 1} and Fragment {pair[1] + 1} "
                    f"(embedding similarity: {fork['embedding_sim']:.2f}, "
                    f"TF-IDF similarity: {fork['tfidf_sim']:.2f}, "
                    f"fork magnitude: {fork['fork_magnitude']:.2f})"
                )
            sections.append("")
            sections.append(
                "**Forks (same topic, different emphasis/vocabulary):**"
            )
            sections.append("\n".join(lines))

        # -- Echoes --
        echoes = analysis.get("echoes", [])
        if echoes:
            lines = []
            for echo in echoes:
                pair = echo["pair"]
                traits = ", ".join(echo.get("shared_traits", []))
                lines.append(
                    f"- Fragment {pair[0] + 1} and Fragment {pair[1] + 1} "
                    f"(embedding similarity: {echo['embedding_sim']:.2f}, "
                    f"sentiment proximity: {echo['sentiment_proximity']:.2f}, "
                    f"shared traits: {traits})"
                )
            sections.append("")
            sections.append(
                "**Echoes (different topics, same voice and feeling):**"
            )
            sections.append("\n".join(lines))

        # -- Shifts --
        shifts = analysis.get("shifts", [])
        if shifts:
            # Include the lexical profile of each shift fragment for context
            lexical_profiles = analysis.get("lexical_profiles", [])
            lines = []
            for idx in shifts:
                profile_desc = ""
                if idx < len(lexical_profiles):
                    lp = lexical_profiles[idx]
                    traits = []
                    if lp.get("question_density", 0) > 0.3:
                        traits.append("highly questioning")
                    if lp.get("hedging_ratio", 0) > 0.3:
                        traits.append("heavily hedged")
                    if lp.get("first_person_ratio", 0) > 0.1:
                        traits.append("personal voice")
                    if lp.get("vocabulary_richness", 0) > 0.9:
                        traits.append("diverse vocabulary")
                    if lp.get("avg_sentence_length", 0) > 25:
                        traits.append("long sentences")
                    elif lp.get("avg_sentence_length", 0) < 8:
                        traits.append("short sentences")
                    if traits:
                        profile_desc = f" ({', '.join(traits)})"
                lines.append(f"- Fragment {idx + 1}{profile_desc}")
            sections.append("")
            sections.append(
                "**Shifts (distinctive voice -- markedly different writing character):**"
            )
            sections.append("\n".join(lines))

        # -- Unclustered --
        unclustered = analysis.get("unclustered", [])
        if unclustered:
            unclustered_text = ", ".join(
                f"Fragment {idx + 1}" for idx in unclustered
            )
            sections.append("")
            sections.append(
                f"**Unclustered (moderate connections, no strong group):** {unclustered_text}"
            )

        sections.append("")
        sections.append(
            "Describe what this map reveals about how these "
            "fragments relate to each other."
        )

        return "\n".join(sections)

    async def narrate(self, analysis: dict) -> str:
        """
        Generate a structural narrative for the fragment map.

        Args:
            analysis: The complete analysis dict from FragmentAnalyser.

        Returns:
            The narrative text.
        """
        user_prompt = self._build_user_prompt(analysis)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=700,
        )

        content = response.choices[0].message.content or ""
        return content.strip()

    async def revise_finding(
        self,
        rule_type: str,
        fragment_a: str,
        fragment_b: str | None,
        original_finding: str,
        scores: dict,
        clarification: str,
        score_deltas: dict | None = None,
    ) -> str:
        """
        Re-narrate a single finding after re-running the pipeline with
        the student's clarification as additional input.

        The signals are re-computed on the combined text (original fragment
        + clarification). Scores may have changed. The narrative references
        what moved and what did not.

        Args:
            rule_type: One of neighbour, stray, rift, fork, echo, shift.
            fragment_a: The first (or only) fragment text.
            fragment_b: The second fragment text (None for stray/shift).
            original_finding: The original narrative for this finding.
            scores: The updated Stage 2 scores after re-running signals.
            clarification: What the student typed.
            score_deltas: Dict of {metric: {old, new, delta}} showing changes.

        Returns:
            The revised narrative text for this finding.
        """
        # Build fragment description
        if fragment_b:
            fragment_desc = (
                f'Fragment A: "{fragment_a}"\n'
                f'Fragment B: "{fragment_b}"'
            )
        else:
            fragment_desc = f'Fragment: "{fragment_a}"'

        # Format scores for the prompt
        score_lines = []
        for k, v in scores.items():
            if isinstance(v, float):
                score_lines.append(f"  {k}: {v:.4f}")
            elif isinstance(v, list):
                score_lines.append(f"  {k}: {', '.join(str(x) for x in v)}")
            else:
                score_lines.append(f"  {k}: {v}")
        scores_text = "\n".join(score_lines) if score_lines else "  (no numeric scores for this finding type)"

        # Format score deltas
        delta_text = ""
        if score_deltas:
            delta_lines = []
            for metric, d in score_deltas.items():
                direction = "increased" if d["delta"] > 0 else "decreased"
                delta_lines.append(f"  {metric}: {d['old']:.3f} → {d['new']:.3f} ({direction} by {abs(d['delta']):.3f})")
            delta_text = "\n\nScore changes after your clarification:\n" + "\n".join(delta_lines)
            delta_text += "\n\nThese changes are real — the tool re-ran its analysis with your clarification included."
        else:
            delta_text = "\n\nScores could not be re-computed for this finding type (requires the full fragment set). The narrative is revised with your context but the original scores stand."

        system_prompt = f"""You are revising a single finding from a fragment analysis tool. The tool has re-run its analysis with the student's clarification included as additional text.

The tool originally found a {rule_type} between these fragments:
{fragment_desc}

The original finding was: {original_finding}

Updated scores (after re-running with clarification):
{scores_text}
{delta_text}

The student's clarification: {clarification}

Revise the narrative for this ONE finding based on the updated scores.

If the scores moved significantly (e.g., embedding similarity increased by >0.1), the clarification made a real difference — the connection the student described is now visible in the combined text. Acknowledge this: "With your clarification included, the semantic distance between these fragments dropped from X to Y. The connection you described — [quote their clarification] — is genuine but was not visible in the original text alone."

If the scores barely moved (<0.05 change), the clarification did not change the tool's reading. Say so honestly: "Even with your clarification, the distance between these fragments remains high (X). The connection you describe may be real, but it is not yet visible in your writing."

2-3 sentences. Be specific. Reference the actual score changes. Quote brief phrases from the student's clarification and fragments. Do not use markdown formatting."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Revise this finding."},
            ],
            temperature=0,
            max_tokens=250,
        )

        content = response.choices[0].message.content or ""
        return content.strip()

    async def narrate_stream(
        self, analysis: dict
    ) -> AsyncGenerator[str, None]:
        """
        Generate narrative with streaming. Yields content chunks.

        Args:
            analysis: The complete analysis dict from FragmentAnalyser.

        Yields:
            String chunks of the narrative as they arrive.
        """
        user_prompt = self._build_user_prompt(analysis)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=700,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
