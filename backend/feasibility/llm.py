"""LLM-powered constraint generation for feasibility checks."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_ollama.llms import OllamaLLM


class ConstraintGenerationError(RuntimeError):
    """Raised when the LLM fails to produce usable constraints."""


class LLMConstraintGenerator:
    """Generate dataset-specific feasibility constraints with an LLM."""

    _PROMPT = """
        You are an expert data scientist designing feasibility rules for
        counterfactual explanations. The goal is to keep suggested changes realistic
        and actionable.

        Dataset: {dataset_name}
        Samples: {row_count}

        Feature summaries:
        {feature_summaries}

        Analyse the features and produce JSON with these keys:
          - "immutable": an array of feature names that should never change.
          - "partially_immutable": an object that maps feature names to
            {{"min": number, "max": number}} or, for categorical features,
            {{"categories": [values]}} describing the limited adjustments
            that are acceptable.
          - "realistic_ranges": an object that maps feature names to
            {{"min": number, "max": number}} describing realistic ranges for
            fully mutable features.

        Use lower-case snake_case for feature names and only reference the
        features listed above. If you are unsure about a feature, omit it.
        Respond with JSON only, no commentary.
        """

    def __init__(self, model_name: str = "llama3.2") -> None:
        self._llm = OllamaLLM(model=model_name)

    def build(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Return structured constraints for the supplied dataset metadata."""

        if not metadata:
            raise ConstraintGenerationError("Missing metadata for constraint generation")

        summaries = metadata.get("feature_summaries", [])
        if not summaries:
            raise ConstraintGenerationError("No feature summaries available for LLM prompt")

        prompt_payload = {
            "dataset_name": metadata.get("dataset_name", "dataset"),
            "row_count": metadata.get("row_count", "unknown"),
            "feature_summaries": self._format_feature_summaries(summaries),
        }

        prompt = self._PROMPT.format(**prompt_payload)
        response = self._llm.invoke(prompt)
        raw_text = response.strip() if isinstance(response, str) else str(response)
        json_blob = self._extract_json(raw_text)

        try:
            return json.loads(json_blob)
        except json.JSONDecodeError as exc:
            raise ConstraintGenerationError("LLM response was not valid JSON") from exc

    @staticmethod
    def _format_feature_summaries(summaries: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for summary in summaries:
            name = summary.get("name", "unknown")
            dtype = summary.get("type", "unknown")
            detail_parts: List[str] = []

            numeric_fields = [
                ("min", summary.get("min")),
                ("max", summary.get("max")),
                ("median", summary.get("median")),
                ("p05", summary.get("p05")),
                ("p95", summary.get("p95")),
            ]
            numeric_details = [
                f"{label}: {value}" for label, value in numeric_fields if value is not None
            ]

            if numeric_details:
                detail_parts.append(", ".join(numeric_details))

            categories = summary.get("top_values")
            if categories:
                detail_parts.append(f"top values: {categories}")

            hints = summary.get("immutability_hint")
            if hints:
                detail_parts.append(f"hint: {hints}")

            detail = "; ".join(detail_parts) if detail_parts else "no stats"
            lines.append(f"- {name} ({dtype}): {detail}")

        return "\n".join(lines)

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ConstraintGenerationError("Could not locate JSON object in LLM response")
        return match.group(0)
