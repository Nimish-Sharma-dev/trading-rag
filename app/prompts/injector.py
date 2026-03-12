"""
app/prompts/injector.py
───────────────────────
Prompt Injection Layer

Takes the retrieved expert prompts and the user query, then assembles
the full context that is sent to the LLM.

The injected context follows this structure:
──────────────────────────────────────────────────────────────────────
[SYSTEM]
You are an expert trading analyst. You must structure your response
according to the frameworks provided below.

=== EXPERT ANALYSIS FRAMEWORKS ===

[Framework 1 — category: technical]
<title>
<content>

[Framework 2 — …]
…
=== END FRAMEWORKS ===

[USER]
<original query>
──────────────────────────────────────────────────────────────────────

The system prompt is deterministic and the frameworks are inserted in
ranked order (highest relevance first).
"""
from __future__ import annotations

from typing import List

from app.retrieval.retriever import RetrievedPrompt


# ─── System prompt template ───────────────────────────────────────────────────

_SYSTEM_TEMPLATE = """\
You are an expert trading and financial analyst with deep knowledge of \
technical analysis, fundamental research, market structure, and risk management.

Your task is to provide a rigorous, structured analysis based on the user's \
query. You MUST follow the analysis frameworks provided below — they define \
the structure and methodology of your response.

Do NOT deviate from the frameworks. Do NOT give generic commentary. \
Be specific, data-driven, and actionable.

{frameworks_block}

Additional instructions:
- Always include a clear "Summary" section at the end.
- If the query involves a specific ticker, analyse it explicitly.
- Use professional financial language throughout.
- Highlight risks alongside opportunities.
"""

_FRAMEWORKS_HEADER = "=== EXPERT ANALYSIS FRAMEWORKS ==="
_FRAMEWORKS_FOOTER = "=== END FRAMEWORKS ==="

_FRAMEWORK_TEMPLATE = """\
[Framework {index} — {category}]
Title: {title}
Tags: {tags}

{content}
"""


# ─── Assembler ────────────────────────────────────────────────────────────────

class PromptInjector:
    """
    Assembles the final (system_prompt, user_message) pair for the LLM.
    """

    def build_context(
        self,
        query: str,
        retrieved_prompts: List[RetrievedPrompt],
        ticker: str | None = None,
        analysis_type: str | None = None,
    ) -> dict:
        """
        Returns a dict with keys:
          system_prompt  — full system instructions with injected frameworks
          user_message   — enriched user query
          metadata       — diagnostic info (scores, ids)
        """
        frameworks_block = self._build_frameworks_block(retrieved_prompts)
        system_prompt = _SYSTEM_TEMPLATE.format(frameworks_block=frameworks_block)

        # Enrich the user message with optional context hints
        user_message = self._build_user_message(query, ticker, analysis_type)

        return {
            "system_prompt": system_prompt,
            "user_message": user_message,
            "metadata": {
                "retrieved_prompt_ids":     [p.id for p in retrieved_prompts],
                "retrieved_prompt_titles":  [p.title for p in retrieved_prompts],
                "cosine_scores":            [p.cosine_score for p in retrieved_prompts],
                "final_scores":             [p.final_score for p in retrieved_prompts],
                "num_frameworks_injected":  len(retrieved_prompts),
            },
        }

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _build_frameworks_block(self, prompts: List[RetrievedPrompt]) -> str:
        if not prompts:
            return (
                f"{_FRAMEWORKS_HEADER}\n"
                "[No specific frameworks retrieved — use general best practices]\n"
                f"{_FRAMEWORKS_FOOTER}"
            )

        parts = [_FRAMEWORKS_HEADER]
        for i, p in enumerate(prompts, start=1):
            parts.append(
                _FRAMEWORK_TEMPLATE.format(
                    index=i,
                    category=p.category,
                    title=p.title,
                    tags=", ".join(p.tags) if p.tags else "—",
                    content=p.content.strip(),
                )
            )
        parts.append(_FRAMEWORKS_FOOTER)
        return "\n".join(parts)

    def _build_user_message(
        self,
        query: str,
        ticker: str | None,
        analysis_type: str | None,
    ) -> str:
        lines = []
        if ticker:
            lines.append(f"Ticker: {ticker.upper()}")
        if analysis_type:
            lines.append(f"Requested analysis type: {analysis_type}")
        lines.append("")
        lines.append(query)
        return "\n".join(lines)
