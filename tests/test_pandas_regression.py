"""Regression tests for PandasEngine optimization."""

from __future__ import annotations

import pandas as pd

from dqflow.column import Column, CrossColumnRule
from dqflow.contract import Contract
from dqflow.engines.pandas import PandasEngine


def test_not_null_validation_regression() -> None:
    """Ensure not_null validation behaves correctly."""

    df = pd.DataFrame(
        {
            "customer_id": [1, 2, None],
        }
    )

    contract = Contract(
        name="not_null_test",
        columns={
            "customer_id": Column(
                dtype=int,
                not_null=True,
            )
        },
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    check = next(
        c for c in result.checks
        if c.name == "not_null:customer_id"
    )

    assert check.passed is False
    assert check.details["null_count"] == 1


def test_min_max_validation_regression() -> None:
    """Ensure min and max validations still work."""

    df = pd.DataFrame(
        {
            "age": [10, 20, 30],
        }
    )

    contract = Contract(
        name="range_test",
        columns={
            "age": Column(
                dtype=int,
                min=18,
                max=65,
            )
        },
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    checks = {
        check.name: check
        for check in result.checks
    }

    assert checks["min:age"].passed is False
    assert checks["max:age"].passed is True


def test_allowed_values_validation_regression() -> None:
    """Ensure allowed values validation works."""

    df = pd.DataFrame(
        {
            "status": ["active", "inactive", "unknown"],
        }
    )

    contract = Contract(
        name="allowed_test",
        columns={
            "status": Column(
                dtype=str,
                allowed=[
                    "active",
                    "inactive",
                ],
            )
        },
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    check = next(
        c for c in result.checks
        if c.name == "allowed:status"
    )

    assert check.passed is False
    assert "unknown" in check.details["invalid_values"]


def test_unique_validation_regression() -> None:
    """Ensure duplicate detection still works."""

    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3],
        }
    )

    contract = Contract(
        name="unique_test",
        columns={
            "id": Column(
                dtype=int,
                unique=True,
            )
        },
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    check = next(
        c for c in result.checks
        if c.name == "unique:id"
    )

    assert check.passed is False
    assert check.details["duplicate_count"] == 2


def test_missing_column_regression() -> None:
    """Ensure missing columns are reported."""

    df = pd.DataFrame(
        {
            "name": ["Alice"],
        }
    )

    contract = Contract(
        name="missing_column_test",
        columns={
            "age": Column(
                dtype=int,
                not_null=True,
            )
        },
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    check = next(
        c for c in result.checks
        if c.name == "column_exists:age"
    )

    assert check.passed is False


def test_stats_cache_regression() -> None:
    """Ensure statistics cache returns expected values."""

    df = pd.DataFrame(
        {
            "value": [1, 2, None, 4],
        }
    )

    cache = PandasEngine()._build_stats_cache(df)

    assert cache["value"]["row_count"] == 4
    assert cache["value"]["unique_count"] == 4
    assert cache["value"]["null_rate"] == 0.25


def test_custom_rule_regression() -> None:
    """Ensure contract rules still evaluate correctly."""

    df = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    contract = Contract(
        name="rule_test",
        columns={
            "value": Column(
                dtype=int,
            )
        },
        rules=[
            "row_count == 3",
        ],
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    check = next(
        c for c in result.checks
        if c.name == "rule:row_count == 3"
    )

    assert check.passed is True


def test_cross_column_rule_regression() -> None:
    """Ensure cross-column validation still works."""

    df = pd.DataFrame(
        {
            "start": [1, 5],
            "end": [2, 3],
        }
    )

    contract = Contract(
        name="cross_column_test",
        columns={
            "start": Column(dtype=int),
            "end": Column(dtype=int),
        },
        cross_column_rules=[
            CrossColumnRule(
                name="start_less_than_end",
                left="start",
                op="<=",
                right="end",
            )
        ],
    )

    result = PandasEngine().validate(
        df,
        contract,
    )

    check = next(
        c for c in result.checks
        if c.name == "cross_column:start_less_than_end"
    )

    assert check.passed is False
    assert check.details["failing_rows"] == 1