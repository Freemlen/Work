"""Prototype for handling parallel promotions without excluding customers.

This module demonstrates a multi-step pipeline that keeps all customers in the
analysis while still correcting for the influence of overlapping promotions.
The prototype uses synthetic data so it can be executed locally without access
to the production warehouse tables referenced in the notebook 3.3 pipeline.

Key ideas implemented below:

1. Build *non-promotional* customer features that will be used in the
   propensity-score model. Promotional metrics are deliberately excluded to
   respect the restriction that PS should not contain campaign signals.
2. Summarise parallel promotion intensity per customer (share of days with a
   crossing, average discount strength, etc.). These statistics are used only in
   weighting and outcome models.
3. Reweight transactions instead of dropping them. The weights decrease the
   influence of days with many overlapping mechanics, but all checks remain in
   the dataset.
4. Estimate uplift via a doubly-robust estimator that combines the propensity
   model with two outcome models (for target and control customers). The outcome
   models consume the cross-promotion statistics so that they capture how
   parallel actions distort the observed spend.

The implementation relies on pandas/numpy/scikit-learn and can be transplanted
into the notebook after swapping the synthetic generators with real tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PropensityModel:
    """Container with the fitted propensity model and feature transformer."""

    model: Pipeline
    feature_columns: Iterable[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return propensity scores for the provided customer-level features."""

        return self.model.predict_proba(features[self.feature_columns])[:, 1]


def build_propensity_model(
    customers: pd.DataFrame, treatment_column: str, *, random_state: int = 42
) -> PropensityModel:
    """Fit a logistic regression on non-promotional columns.

    Args:
        customers: DataFrame with customer-level attributes. It must not include
            promotional variables; in production these features can be created
            from socio-demographics, channel mix, basket structure, etc.
        treatment_column: Name of the binary indicator that marks membership in
            the target audience.
        random_state: Seed used for deterministic logistic regression.

    Returns:
        PropensityModel ready to score arbitrary customer profiles.
    """

    feature_columns = [col for col in customers.columns if col != treatment_column]

    num_cols = [c for c in feature_columns if customers[c].dtype != "object"]
    cat_cols = [c for c in feature_columns if customers[c].dtype == "object"]

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("features", transformer),
            (
                "clf",
                LogisticRegression(
                    penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=random_state
                ),
            ),
        ]
    )

    model.fit(customers[feature_columns], customers[treatment_column])
    return PropensityModel(model=model, feature_columns=feature_columns)


def summarise_cross_promotions(days_cross: pd.DataFrame) -> pd.DataFrame:
    """Aggregate information about parallel promotions per customer.

    Args:
        days_cross: DataFrame with the following columns:
            - customer_id: unique customer identifier.
            - date: date of the transaction day.
            - is_cross: binary flag (1 if the target action crossed with another
              mechanic on that day).
            - discount_strength: intensity of the strongest overlapping
              promotion on that day (e.g. absolute discount percentage).

    Returns:
        DataFrame with customer-level statistics describing the share and
        intensity of crossing days. These metrics are suitable for downstream
        weighting and outcome models.
    """

    summary = (
        days_cross.groupby("customer_id").apply(
            lambda df: pd.Series(
                {
                    "share_cross_days": df["is_cross"].mean(),
                    "avg_discount_cross": df.loc[df["is_cross"] == 1, "discount_strength"].mean()
                    if (df["is_cross"] == 1).any()
                    else 0.0,
                    "max_discount_cross": df["discount_strength"].max(),
                }
            )
        )
    )

    # Fill missing values in case a customer never had a crossing.
    summary = summary.fillna(0.0)
    return summary.reset_index()


def compute_transaction_weights(
    transactions: pd.DataFrame, cross_summary: pd.DataFrame, *, alpha: float = 1.2
) -> pd.Series:
    """Create weights that downplay high-intensity crossing days.

    The weight function keeps all transactions but scales them based on the
    customer's share of crossing days and the observed discount intensity.
    Larger values of ``alpha`` produce a stronger penalisation.
    """

    joined = transactions.merge(cross_summary, on="customer_id", how="left").fillna(0.0)
    penalties = 1 + alpha * joined["share_cross_days"] * (1 + joined["avg_discount_cross"])
    weights = 1 / penalties
    return pd.Series(weights, index=transactions.index, name="weight")


def fit_outcome_models(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    cross_summary: pd.DataFrame,
    treatment_column: str,
    target_column: str,
) -> Tuple[Pipeline, Pipeline]:
    """Train separate outcome models for treated and control customers.

    The outcome models receive the non-promotional features together with the
    cross-promotion summary so that they can capture how overlapping mechanics
    distort the observed spend.
    """

    data = features.merge(cross_summary, on="customer_id", how="left").fillna(0.0)
    data = data.merge(
        transactions.groupby("customer_id")[target_column].mean().rename("avg_spend"),
        left_on="customer_id",
        right_index=True,
        how="left",
    ).fillna(0.0)

    outcome_features = [c for c in data.columns if c not in {"customer_id", "avg_spend"}]

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), outcome_features),
        ],
        remainder="drop",
    )

    treated_mask = features[treatment_column] == 1
    control_mask = ~treated_mask

    regressor_t = Pipeline([
        ("features", transformer),
        ("reg", LassoCV(cv=5, random_state=0)),
    ])
    regressor_c = Pipeline([
        ("features", transformer),
        ("reg", LassoCV(cv=5, random_state=0)),
    ])

    regressor_t.fit(data.loc[treated_mask, outcome_features], data.loc[treated_mask, "avg_spend"])
    regressor_c.fit(data.loc[control_mask, outcome_features], data.loc[control_mask, "avg_spend"])
    return regressor_t, regressor_c


def doubly_robust_effect(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    cross_summary: pd.DataFrame,
    propensity: PropensityModel,
    *,
    treatment_column: str,
    target_column: str,
) -> pd.DataFrame:
    """Compute customer-level uplift using a doubly-robust estimator."""

    weights = compute_transaction_weights(transactions, cross_summary)
    weighted_avg = (
        transactions.assign(weight=weights)
        .groupby("customer_id")
        .apply(lambda df: np.average(df[target_column], weights=df["weight"]))
        .rename("weighted_outcome")
    )

    # Fit outcome models on weighted outcomes.
    outcome_features = features.merge(cross_summary, on="customer_id", how="left").fillna(0.0)
    outcome_features = outcome_features.join(weighted_avg, on="customer_id").fillna(0.0)

    treated_mask = outcome_features[treatment_column] == 1
    control_mask = ~treated_mask

    regressors = fit_outcome_models(
        transactions.assign(weight=weights, spend=transactions[target_column] * weights),
        outcome_features[[c for c in outcome_features.columns if c != "weighted_outcome"]],
        cross_summary,
        treatment_column,
        target_column="spend",
    )
    reg_t, reg_c = regressors

    mu1 = reg_t.predict(outcome_features.drop(columns=["weighted_outcome"]))
    mu0 = reg_c.predict(outcome_features.drop(columns=["weighted_outcome"]))

    propensity_scores = propensity.predict(outcome_features)

    observed = outcome_features["weighted_outcome"].to_numpy()
    treatment = outcome_features[treatment_column].to_numpy()

    tau_dr = (
        mu1 - mu0
        + treatment * (observed - mu1) / np.clip(propensity_scores, 1e-3, 1 - 1e-3)
        - (1 - treatment) * (observed - mu0) / np.clip(1 - propensity_scores, 1e-3, 1 - 1e-3)
    )

    return pd.DataFrame(
        {
            "customer_id": outcome_features["customer_id"],
            "uplift": tau_dr,
            "propensity": propensity_scores,
            "weighted_outcome": observed,
        }
    )


def generate_synthetic_inputs(n_customers: int = 500, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create synthetic customers, transactions, and day-level crossing info."""

    rng = np.random.default_rng(seed)
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(n_customers),
            "treatment": rng.integers(0, 2, size=n_customers),
            "age": rng.normal(40, 8, size=n_customers),
            "is_online": rng.integers(0, 2, size=n_customers).astype("category"),
            "avg_baseline_bill": rng.gamma(shape=2, scale=400, size=n_customers),
        }
    )

    days = np.repeat(customers["customer_id"].to_numpy(), repeats=7)
    date_index = pd.date_range("2024-01-01", periods=7)
    date_index = np.tile(date_index, n_customers)

    days_cross = pd.DataFrame(
        {
            "customer_id": days,
            "date": date_index,
            "is_cross": rng.binomial(1, p=0.2, size=len(days)),
            "discount_strength": rng.uniform(0, 0.4, size=len(days)),
        }
    )

    transactions = pd.DataFrame(
        {
            "customer_id": np.repeat(customers["customer_id"], repeats=3),
            "transaction_id": np.arange(n_customers * 3),
            "spend": rng.gamma(shape=2, scale=200, size=n_customers * 3),
        }
    )

    return customers, transactions, days_cross


def demo_pipeline() -> pd.DataFrame:
    """Run the complete prototype pipeline on synthetic data."""

    customers, transactions, days_cross = generate_synthetic_inputs()
    cross_summary = summarise_cross_promotions(days_cross)
    propensity = build_propensity_model(customers, treatment_column="treatment")

    uplift = doubly_robust_effect(
        transactions,
        features=customers.rename(columns={"treatment": "treatment"}),
        cross_summary=cross_summary,
        propensity=propensity,
        treatment_column="treatment",
        target_column="spend",
    )

    uplift = uplift.merge(cross_summary, on="customer_id", how="left").fillna(0.0)

    # Calculate diagnostic error between treated and control predictions.
    treated = uplift[uplift["propensity"] > 0.5]
    control = uplift[uplift["propensity"] <= 0.5]
    rmse = mean_squared_error(treated["weighted_outcome"], control["weighted_outcome"], squared=False)

    return uplift.assign(rmse_diagnostic=rmse)


if __name__ == "__main__":
    result = demo_pipeline()
    pd.set_option("display.max_columns", None)
    print(result.head())
