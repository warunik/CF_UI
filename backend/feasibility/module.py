"""Feasibility enforcement utilities for counterfactual explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import re

import numpy as np

from .llm import ConstraintGenerationError, LLMConstraintGenerator


@dataclass
class FeatureCheckResult:
    """Holds feasibility information for a single feature."""

    feature: str
    feasible: bool
    reason: str = ""
    constraint_type: str = "mutable"
    adjusted_value: Any = None
    original_value: Any = None
    operator: Optional[str] = None


class FeasibilityModule:
    """Validate and refine counterfactual suggestions.

    The module enforces dataset-specific constraints that capture immutable
    features, acceptable value ranges, and categorical limits.  Each dataset is
    described through a set of immutable features, partially immutable
    constraints (typically value ranges), and realistic ranges that represent
    plausible values observed in the training corpus.
    """

    def __init__(self, generator: Optional[LLMConstraintGenerator] = None) -> None:
        self.feature_constraints = self._build_default_constraints()
        self._generator = generator

    def ensure_constraints(
        self,
        dataset: str,
        metadata: Optional[Dict[str, Any]],
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Ensure that feasibility constraints exist for ``dataset``."""

        key = dataset.lower()

        if not force and key in self.feature_constraints:
            return self.feature_constraints[key]

        if metadata is None and not force:
            return self.feature_constraints.get(key)

        payload: Optional[Dict[str, Any]] = None
        generator = self._generator or LLMConstraintGenerator()
        self._generator = generator

        if metadata is not None:
            try:
                payload = generator.build(metadata)
            except ConstraintGenerationError:
                payload = None

        normalized = self._normalize_generated_constraints(payload)
        if normalized:
            existing = self.feature_constraints.get(key)
            merged = self._merge_constraints(existing, normalized)
            merged["source"] = "llm"
            self.feature_constraints[key] = merged
            return self.feature_constraints[key]

        fallback = self._fallback_constraints(metadata)
        if fallback:
            existing = self.feature_constraints.get(key)
            merged = self._merge_constraints(existing, fallback)
            merged["source"] = "data"
            self.feature_constraints[key] = merged
            return self.feature_constraints[key]

        return self.feature_constraints.get(key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enforce(
        self,
        dataset: str,
        original_instance: Dict[str, Any],
        original_prediction: int,
        raw_changes: Iterable[Dict[str, Any]],
        predict_fn,
    ) -> Dict[str, Any]:
        """Apply feasibility checks to the supplied counterfactual changes.

        Parameters
        ----------
        dataset:
            Dataset identifier (as used in ``DATASETS`` configuration).
        original_instance:
            Mapping of feature names to their original values.
        original_prediction:
            The predicted class for the original instance.  Used to verify that
            the refined counterfactual still flips the model's decision.
        raw_changes:
            Iterable with the structured counterfactual suggestions produced by
            the explainer.
        predict_fn:
            Callable that receives a dictionary of feature values and returns
            the predicted class label from the ML model.
        """

        constraints = self.ensure_constraints(dataset, metadata=None)
        final_instance = dict(original_instance)
        sanitized_changes: List[Dict[str, Any]] = []
        adjustments: List[str] = []
        violations: List[str] = []

        if not raw_changes:
            final_prediction = predict_fn(final_instance)
            report = self._build_report(True, False, adjustments, violations)
            return {
                "changes": [],
                "feasible": True,
                "valid_counterfactual": final_prediction != original_prediction,
                "adjustments": adjustments,
                "violations": violations,
                "final_instance": final_instance,
                "final_prediction": final_prediction,
                "report": report,
            }

        for change in raw_changes:
            feature_label = change.get("feature", "")
            operator = self._format_operator(change.get("operator"))
            target_value = change.get("new")

            match_key, original_value = self._match_feature(final_instance, feature_label)
            normalized_feature = self._normalize_feature(feature_label)

            if match_key is None:
                violations.append(
                    f"Feature '{feature_label}' is not part of the original instance and was skipped."
                )
                continue

            # Attempt to coerce the proposed value to a numeric type when the
            # original value is numeric.  This makes range comparisons robust.
            coerced_target = self._coerce_like(original_value, target_value)
            check = self._check_feature(
                constraints=constraints,
                feature_key=normalized_feature,
                operator=operator,
                target_value=coerced_target,
                original_value=original_value,
            )

            if not check.feasible:
                violations.append(check.reason or f"{feature_label} violates feasibility constraints.")
                continue

            adjusted_value = check.adjusted_value if check.adjusted_value is not None else coerced_target
            cast_adjusted = self._coerce_like(original_value, adjusted_value)

            if constraints and adjusted_value != coerced_target:
                adjustments.append(
                    f"Adjusted {feature_label} to {cast_adjusted} to satisfy feasibility limits."
                )

            final_instance[match_key] = cast_adjusted

            sanitized_changes.append(
                {
                    "feature": feature_label,
                    "current": original_value,
                    "new": cast_adjusted,
                    "operator": operator,
                }
            )

        final_prediction = predict_fn(final_instance)
        is_feasible = not violations
        valid_counterfactual = final_prediction != original_prediction
        report = self._build_report(is_feasible, valid_counterfactual, adjustments, violations)

        return {
            "changes": sanitized_changes,
            "feasible": is_feasible,
            "valid_counterfactual": valid_counterfactual,
            "adjustments": adjustments,
            "violations": violations,
            "final_instance": final_instance,
            "final_prediction": final_prediction,
            "report": report,
        }

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------
    def _check_feature(
        self,
        constraints: Optional[Dict[str, Any]],
        feature_key: str,
        operator: Optional[str],
        target_value: Any,
        original_value: Any,
    ) -> FeatureCheckResult:
        result = FeatureCheckResult(
            feature=feature_key,
            feasible=True,
            operator=operator,
            original_value=original_value,
            adjusted_value=target_value,
        )

        if constraints is None:
            return result

        immutable = constraints.get("immutable", set())
        partially_immutable = constraints.get("partially_immutable", {})
        realistic_ranges = constraints.get("realistic_ranges", {})

        if feature_key in immutable and target_value != original_value:
            result.feasible = False
            result.constraint_type = "immutable"
            result.reason = f"{self._display_feature(feature_key)} is immutable and cannot be changed."
            return result

        # Helper to enforce range limits.
        def clamp(value: float, bounds: Tuple[Optional[float], Optional[float]]):
            lower, upper = bounds
            new_value = value
            if lower is not None:
                new_value = max(lower, new_value)
            if upper is not None:
                new_value = min(upper, new_value)
            return new_value

        if feature_key in partially_immutable:
            bounds = partially_immutable[feature_key]
            if isinstance(bounds, tuple):
                numeric_target = self._ensure_numeric(target_value)
                if numeric_target is None:
                    result.feasible = False
                    result.constraint_type = "partially_immutable"
                    result.reason = (
                        f"{self._display_feature(feature_key)} must be numeric to evaluate feasibility."
                    )
                    return result

                clamped_value = clamp(numeric_target, bounds)
                if not np.isclose(clamped_value, numeric_target):
                    result.adjusted_value = clamped_value
            elif isinstance(bounds, (list, set)) and bounds:
                if target_value not in bounds:
                    result.feasible = False
                    result.constraint_type = "partially_immutable"
                    result.reason = (
                        f"{self._display_feature(feature_key)} must be one of {sorted(bounds)}."
                    )
                    return result

        if feature_key in realistic_ranges:
            bounds = realistic_ranges[feature_key]
            numeric_target = self._ensure_numeric(result.adjusted_value)
            if numeric_target is None:
                result.feasible = False
                result.constraint_type = "range"
                result.reason = f"{self._display_feature(feature_key)} must be numeric."
                return result

            clamped_value = clamp(numeric_target, bounds)
            if not np.isclose(clamped_value, numeric_target):
                result.adjusted_value = clamped_value

        # If the operator indicates the condition is already satisfied, avoid
        # suggesting the change.
        if operator and self._violates_operator(operator, original_value, result.adjusted_value):
            result.feasible = False
            result.reason = (
                f"{self._display_feature(feature_key)} already satisfies the "
                f"condition '{operator} {result.adjusted_value}'."
            )

        return result

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _normalize_generated_constraints(
        self, payload: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not payload:
            return None

        immutable = set()
        partially: Dict[str, Any] = {}
        realistic: Dict[str, Any] = {}

        for feature in payload.get("immutable", []) or []:
            immutable.add(self._normalize_feature(str(feature)))

        for collection_key in ("partially_immutable", "categorical_limits"):
            for feature, value in (payload.get(collection_key) or {}).items():
                parsed = self._parse_constraint_value(value)
                if parsed is not None:
                    partially[self._normalize_feature(str(feature))] = parsed

        for feature, value in (payload.get("realistic_ranges") or {}).items():
            parsed = self._parse_constraint_value(value)
            if parsed is not None:
                realistic[self._normalize_feature(str(feature))] = parsed

        if not any([immutable, partially, realistic]):
            return None

        return {
            "immutable": immutable,
            "partially_immutable": partially,
            "realistic_ranges": realistic,
        }

    @staticmethod
    def _merge_constraints(
        base: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        merged = {
            "immutable": set(),
            "partially_immutable": {},
            "realistic_ranges": {},
        }

        if base:
            merged["immutable"] = set(base.get("immutable", set()))
            merged["partially_immutable"] = dict(base.get("partially_immutable", {}))
            merged["realistic_ranges"] = dict(base.get("realistic_ranges", {}))

        if new:
            if new.get("immutable"):
                merged["immutable"].update(new["immutable"])
            if new.get("partially_immutable"):
                merged["partially_immutable"].update(new["partially_immutable"])
            if new.get("realistic_ranges"):
                merged["realistic_ranges"].update(new["realistic_ranges"])
            if "source" in new:
                merged["source"] = new["source"]

        return merged

    def _parse_constraint_value(self, value: Any) -> Optional[Any]:
        if value is None:
            return None

        if isinstance(value, dict):
            if "min" in value or "max" in value:
                bounds = (
                    self._ensure_numeric(value.get("min")),
                    self._ensure_numeric(value.get("max")),
                )
                return bounds if any(v is not None for v in bounds) else None
            if "range" in value:
                return self._parse_constraint_value(value["range"])
            if "categories" in value:
                categories = [item for item in value["categories"] if item is not None]
                return categories or None
            if "values" in value:
                values = [item for item in value["values"] if item is not None]
                return values or None

        if isinstance(value, (list, tuple)):
            if len(value) >= 2 and all(self._ensure_numeric(v) is not None for v in value[:2]):
                bounds = (
                    self._ensure_numeric(value[0]),
                    self._ensure_numeric(value[1]),
                )
                return bounds if any(v is not None for v in bounds) else None
            cleaned = [item for item in value if item is not None]
            return cleaned or None

        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = self._ensure_numeric(value)
            if numeric is None:
                return None
            return (numeric, numeric)

        if isinstance(value, str):
            range_tuple = self._parse_range_string(value)
            if range_tuple is not None:
                return range_tuple
            parts = [item.strip() for item in value.split(",") if item.strip()]
            if len(parts) > 1:
                return parts

        return None

    @staticmethod
    def _parse_range_string(text: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
        range_match = re.match(r"\s*(?P<low>-?\d+\.?\d*)\s*[-–]\s*(?P<high>-?\d+\.?\d*)\s*", text)
        if range_match:
            low = range_match.group("low")
            high = range_match.group("high")
            bounds = (
                FeasibilityModule._ensure_numeric(low),
                FeasibilityModule._ensure_numeric(high),
            )
            return bounds if any(v is not None for v in bounds) else None

        ge_match = re.match(r"\s*(>=|>\=)\s*(-?\d+\.?\d*)\s*", text)
        if ge_match:
            value = FeasibilityModule._ensure_numeric(ge_match.group(2))
            return (value, None) if value is not None else None

        le_match = re.match(r"\s*(<=|<\=)\s*(-?\d+\.?\d*)\s*", text)
        if le_match:
            value = FeasibilityModule._ensure_numeric(le_match.group(2))
            return (None, value) if value is not None else None

        return None

    def _fallback_constraints(self, metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not metadata:
            return None

        summaries = metadata.get("feature_summaries") or []
        if not summaries:
            return None

        realistic: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        partially: Dict[str, Any] = {}
        immutables: Set[str] = set()

        for summary in summaries:
            feature_name = self._normalize_feature(str(summary.get("name", "")))
            if not feature_name:
                continue

            feature_type = str(summary.get("type", "")).lower()
            if feature_type == "numeric":
                lower = self._ensure_numeric(summary.get("p05"))
                upper = self._ensure_numeric(summary.get("p95"))
                if lower is None:
                    lower = self._ensure_numeric(summary.get("min"))
                if upper is None:
                    upper = self._ensure_numeric(summary.get("max"))
                if lower is not None or upper is not None:
                    realistic[feature_name] = (lower, upper)
            else:
                top_values = summary.get("top_values") or []
                unique_count = summary.get("unique_count")
                if top_values and unique_count and unique_count <= 8:
                    partially[feature_name] = list(dict.fromkeys(top_values))

            hint = summary.get("immutability_hint")
            if hint:
                immutables.add(feature_name)

        if not any([immutables, partially, realistic]):
            return None

        return {
            "immutable": immutables,
            "partially_immutable": partially,
            "realistic_ranges": realistic,
        }

    @staticmethod
    def _normalize_feature(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    @staticmethod
    def _display_feature(name: str) -> str:
        return name.replace("_", " ").title()

    @staticmethod
    def _format_operator(operator: Any) -> Optional[str]:
        if operator is None:
            return None
        if hasattr(operator, "value"):
            return str(operator.value)
        if hasattr(operator, "name"):
            return str(operator.name)
        return str(operator)

    def _match_feature(self, instance: Dict[str, Any], feature_label: str) -> Tuple[Optional[str], Any]:
        normalized = self._normalize_feature(feature_label)
        for key, value in instance.items():
            if self._normalize_feature(str(key)) == normalized:
                return key, value
        return None, None

    @staticmethod
    def _ensure_numeric(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(numeric):
            return None
        return numeric

    @staticmethod
    def _coerce_like(reference: Any, value: Any) -> Any:
        if reference is None or value is None:
            return value

        # Preserve booleans when feasible.
        if isinstance(reference, (bool, np.bool_)):
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes"}:
                    return True
                if lowered in {"false", "0", "no"}:
                    return False
            return bool(value)

        if isinstance(reference, (int, np.integer)) and not isinstance(reference, (bool, np.bool_)):
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return reference

        if isinstance(reference, (float, np.floating)):
            try:
                return float(value)
            except (TypeError, ValueError):
                return reference

        return value

    @staticmethod
    def _violates_operator(operator: str, original_value: Any, target_value: Any) -> bool:
        numeric_original = FeasibilityModule._ensure_numeric(original_value)
        numeric_target = FeasibilityModule._ensure_numeric(target_value)
        if numeric_original is None or numeric_target is None:
            return False

        if operator in {"<=", "le", "LE"}:
            return numeric_original <= numeric_target
        if operator in {"<", "lt", "LT"}:
            return numeric_original < numeric_target
        if operator in {">=", "ge", "GE"}:
            return numeric_original >= numeric_target
        if operator in {">", "gt", "GT"}:
            return numeric_original > numeric_target
        return False

    @staticmethod
    def _build_report(
        feasible: bool,
        valid_counterfactual: bool,
        adjustments: List[str],
        violations: List[str],
    ) -> str:
        status = "Feasible counterfactual generated." if feasible else "Counterfactual violated feasibility constraints."
        if not valid_counterfactual:
            status += " However, the refined instance does not change the model's prediction."

        lines = [status]
        if adjustments:
            lines.append("Adjustments applied:")
            lines.extend(f"- {msg}" for msg in adjustments)
        if violations:
            lines.append("Violations:")
            lines.extend(f"- {msg}" for msg in violations)
        return "\n".join(lines)

    @staticmethod
    def _build_default_constraints() -> Dict[str, Dict[str, Any]]:
        return {
            "heart": {
                "immutable": {"age", "sex"},
                "partially_immutable": {
                    "ca": (0, 4),
                    "thal": (0, 3),
                    "cp": (0, 3),
                    "slope": (0, 2),
                    "restecg": (0, 2),
                },
                "realistic_ranges": {
                    "age": (29, 77),
                    "trestbps": (94, 200),
                    "chol": (126, 564),
                    "thalach": (71, 202),
                    "oldpeak": (0, 6.2),
                    "fbs": (0, 1),
                    "exang": (0, 1),
                    "sex": (0, 1),
                    "ca": (0, 3),
                    "thal": (0, 3),
                    "cp": (0, 3),
                    "restecg": (0, 2),
                    "slope": (0, 2),
                },
            },
            "diabetes": {
                "immutable": {"age"},
                "partially_immutable": {
                    "pregnancies": (0, 17),
                    "age": (21, 81),
                },
                "realistic_ranges": {
                    "pregnancies": (0, 17),
                    "glucose": (0, 199),
                    "bloodpressure": (0, 122),
                    "skinthickness": (0, 99),
                    "insulin": (0, 846),
                    "bmi": (0, 67.1),
                    "diabetespedigreefunction": (0.08, 2.42),
                    "age": (21, 81),
                },
            },
            "german": {
                "immutable": {"age", "personalstatussex", "foreignworker"},
                "partially_immutable": {
                    "age": (19, 75),
                    "durationinmonth": (4, 72),
                    "creditamount": (250, 18424),
                    "presentressince": (1, 4),
                },
                "realistic_ranges": {
                    "age": (19, 75),
                    "durationinmonth": (4, 72),
                    "creditamount": (250, 18424),
                    "installmentasincomeperc": (1, 4),
                    "presentressince": (1, 4),
                    "creditsthisbank": (1, 4),
                    "peopleundermaintenance": (1, 2),
                },
            },
        }

