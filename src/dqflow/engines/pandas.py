"""Pandas validation engine."""

from __future__ import annotations

import operator as _op
from collections.abc import Callable
from typing import Any

import pandas as pd

from dqflow.column import Column, CrossColumnRule
from dqflow.contract import Contract
from dqflow.engines.base import Engine
from dqflow.result import CheckResult, ValidationResult

_OPS: dict[str, Callable[[Any, Any], Any]] = {
    ">=": _op.ge,
    "<=": _op.le,
    ">": _op.gt,
    "<": _op.lt,
    "==": _op.eq,
    "!=": _op.ne,
}


class PandasEngine(Engine):
    """Validation engine for pandas DataFrames with reduced redundant scans."""

    def validate(
        self,
        data: pd.DataFrame,
        contract: Contract,
        **kwargs: Any,
    ) -> ValidationResult:
        df = data
        result = ValidationResult(contract_name=contract.name)
        cache = self._build_stats_cache(df)

        _ = kwargs.get("parallel", False)
        _ = kwargs.get("max_workers")

        existing = set(df.columns)

        for col_name in contract.columns:
            exists = col_name in existing
            result.checks.append(
                CheckResult(
                    name=f"column_exists:{col_name}",
                    passed=exists,
                    message="" if exists else f"Column '{col_name}' not found in DataFrame",
                )
            )

        for col_name, col_def in contract.columns.items():
            if col_name in existing:
                result.checks.extend(self._validate_column(df[col_name], col_name, col_def))

        result.checks.extend(
            self._evaluate_rule(df, rule, cache) for rule in contract.rules
        )
        result.checks.extend(
            self._evaluate_cross_column_rule(df, rule)
            for rule in contract.cross_column_rules
        )
        return result

    def _validate_column(
        self,
        series: pd.Series,
        col_name: str,
        col_def: Column,
    ) -> list[CheckResult]:
        checks: list[CheckResult] = []

        null_mask = series.isna()
        null_count = int(null_mask.sum())
        non_null = series[~null_mask]

        min_val = series.min() if col_def.min is not None else None
        max_val = series.max() if col_def.max is not None else None
        unique_values = set(non_null.unique()) if col_def.allowed is not None else None
        duplicate_count = int(series.duplicated(keep=False).sum()) if col_def.unique else 0

        if col_def.not_null:
            checks.append(CheckResult(
                name=f"not_null:{col_name}",
                passed=null_count == 0,
                message=f"Found {null_count} null values" if null_count else "",
                details={"null_count": null_count},
            ))

        if col_def.min is not None:
            passed = pd.isna(min_val) or min_val >= col_def.min
            checks.append(CheckResult(
                name=f"min:{col_name}",
                passed=bool(passed),
                message="" if passed else f"Minimum value {min_val} is below {col_def.min}",
                details={"actual_min": float(min_val) if pd.notna(min_val) else None},
            ))

        if col_def.max is not None:
            passed = pd.isna(max_val) or max_val <= col_def.max
            checks.append(CheckResult(
                name=f"max:{col_name}",
                passed=bool(passed),
                message="" if passed else f"Maximum value {max_val} exceeds {col_def.max}",
                details={"actual_max": float(max_val) if pd.notna(max_val) else None},
            ))

        if col_def.allowed is not None:
            invalid = unique_values - set(col_def.allowed)
            checks.append(CheckResult(
                name=f"allowed:{col_name}",
                passed=not invalid,
                message="" if not invalid else f"Found invalid values: {invalid}",
                details={"invalid_values": list(invalid)},
            ))

        if col_def.unique:
            checks.append(CheckResult(
                name=f"unique:{col_name}",
                passed=duplicate_count == 0,
                message="" if duplicate_count == 0 else f"Found {duplicate_count} duplicate values",
                details={"duplicate_count": duplicate_count},
            ))

        return checks

    def _build_stats_cache(self, df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
        row_count = len(df)
        return {
            col: {
                "null_rate": float(df[col].isna().mean()),
                "unique_count": int(df[col].nunique(dropna=False)),
                "row_count": row_count,
            }
            for col in df.columns
        }

    def _evaluate_rule(self, df: pd.DataFrame, rule: str, cache):
        try:
            context = {
                "row_count": len(df),
                "null_rate": lambda c: cache.get(c, {}).get("null_rate", 0),
                "unique_count": lambda c: cache.get(c, {}).get("unique_count", 0),
            }
            result = eval(rule, {"__builtins__": {}}, context)
            return CheckResult(name=f"rule:{rule}", passed=bool(result),
                               message="" if result else f"Rule '{rule}' failed")
        except Exception as e:
            return CheckResult(name=f"rule:{rule}", passed=False,
                               message=f"Failed to evaluate rule: {e}")

    def _evaluate_cross_column_rule(self, df: pd.DataFrame, rule: CrossColumnRule):
        try:
            if rule.check is not None:
                mask = rule.check(df)
            else:
                assert rule.left is not None and rule.op is not None
                right = df[rule.right] if isinstance(rule.right, str) and rule.right in df.columns else rule.right
                mask = _OPS[rule.op](df[rule.left], right)
            failing = int((~mask).sum())
            return CheckResult(
                name=f"cross_column:{rule.name}",
                passed=failing == 0,
                message="" if failing == 0 else rule.error_message,
                details={"failing_rows": failing},
            )
        except Exception as e:
            return CheckResult(
                name=f"cross_column:{rule.name}",
                passed=False,
                message=f"Failed to evaluate cross-column rule '{rule.name}': {e}",
            )
