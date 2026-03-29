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
For strays (isolates): note their distance without judging. A stray fragment is not a problem -- it is a fact.
For rifts (tensions): describe the pull in both directions. Name the emotional opposition. Do not resolve the tension.
For forks: describe the shared topic and the different emphasis. Name the vocabulary gap.
For echoes (resonances): describe the shared voice across different topics. Name what the fragments share that is not topic.
For shifts (distinctive voices): name what changes. "This fragment asks where the others declare."

Do not evaluate, prescribe, or suggest. Do not introduce findings the rules did not produce. Tone: observational. Length: 2-4 sentences for 3-5 fragments, up to 10 for 15-20 fragments."""


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
            temperature=0.5,
            max_tokens=500,
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
            temperature=0.5,
            max_tokens=500,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
