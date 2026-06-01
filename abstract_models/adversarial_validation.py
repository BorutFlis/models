import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score


def adversarial_validation_until_auc_05(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    pipeline,
    target_auc: float = 0.50,
    tolerance: float = 0.01,
    max_iter: int | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Iteratively run adversarial validation using a provided sklearn Pipeline
    that already contains preprocessing (including missing value handling)
    and the classifier.

    The function removes one original feature at a time, using the feature
    importance from the final model inside the pipeline, until adversarial
    ROC-AUC is close to 0.5.

    Parameters
    ----------
    X_train : pd.DataFrame
        Original training features.
    X_test : pd.DataFrame
        Original test features.
    pipeline : sklearn Pipeline
        A fitted-able sklearn pipeline that includes preprocessing and model.
        Example: ColumnTransformer + RandomForestClassifier.
    target_auc : float, default=0.50
        Desired adversarial ROC-AUC.
    tolerance : float, default=0.01
        Stop when abs(auc - target_auc) <= tolerance.
    max_iter : int or None, default=None
        Maximum number of feature-removal iterations.
    n_splits : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Used for CV only.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    dict
        {
            "final_auc": float,
            "history": list[dict],
            "removed_features": list[str],
            "selected_features": list[str],
            "best_pipeline": fitted sklearn Pipeline,
            "adversarial_X": pd.DataFrame,
            "adversarial_y": np.ndarray
        }

    Notes
    -----
    - This assumes the final estimator inside the pipeline exposes
      `feature_importances_` after fitting, e.g. RandomForestClassifier.
    - Feature removal happens at the ORIGINAL COLUMN level, not transformed
      feature level.
    """

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    common_cols = [c for c in X_train.columns if c in X_test.columns]
    if not common_cols:
        raise ValueError("X_train and X_test have no common columns.")

    X_train = X_train[common_cols].copy()
    X_test = X_test[common_cols].copy()

    # Adversarial dataset: train=0, test=1
    X_adv = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_adv = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_test), dtype=int),
    ])

    current_features = common_cols.copy()
    removed_features = []
    history = []

    if max_iter is None:
        max_iter = len(current_features)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    best_auc = None
    best_features = current_features.copy()
    best_pipeline = None

    for iteration in range(max_iter):
        if not current_features:
            break

        X_cur = X_adv[current_features]
        pipe = clone(pipeline)

        auc_scores = cross_val_score(
            pipe,
            X_cur,
            y_adv,
            scoring="roc_auc",
            cv=cv,
            n_jobs=None,  # keep pipeline compatibility safe
        )
        auc_mean = float(np.mean(auc_scores))
        auc_std = float(np.std(auc_scores))

        # Fit on full adversarial dataset to inspect feature importances
        pipe.fit(X_cur, y_adv)

        history.append({
            "iteration": iteration,
            "n_features": len(current_features),
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "removed_feature": removed_features[-1] if removed_features else None,
            "current_features": current_features.copy(),
        })

        if verbose:
            print(
                f"iter={iteration:02d} | "
                f"n_features={len(current_features):4d} | "
                f"auc={auc_mean:.5f} ± {auc_std:.5f}"
            )

        if best_auc is None or abs(auc_mean - target_auc) < abs(best_auc - target_auc):
            best_auc = auc_mean
            best_features = current_features.copy()
            best_pipeline = pipe

        if abs(auc_mean - target_auc) <= tolerance:
            return {
                "final_auc": auc_mean,
                "history": history,
                "removed_features": removed_features,
                "selected_features": current_features,
                "best_pipeline": pipe,
                "adversarial_X": X_cur,
                "adversarial_y": y_adv,
            }

        if auc_mean < target_auc:
            break

        # Get fitted final estimator from pipeline
        final_estimator = pipe.steps[-1][1]

        if not hasattr(final_estimator, "feature_importances_"):
            raise ValueError(
                "The final estimator in the pipeline does not expose "
                "`feature_importances_`."
            )

        # Feature importance mapping:
        # We remove ORIGINAL columns, so we need importance at original-column level.
        # If preprocessing expands columns (e.g. one-hot encoding), we aggregate
        # transformed importances back to original input columns when possible.
        importances_by_input_feature = _extract_input_feature_importance(pipe, current_features)

        worst_feature = importances_by_input_feature.sort_values(ascending=False).index[0]
        current_features.remove(worst_feature)
        removed_features.append(worst_feature)

    return {
        "final_auc": best_auc,
        "history": history,
        "removed_features": removed_features,
        "selected_features": best_features,
        "best_pipeline": best_pipeline,
        "adversarial_X": X_adv[best_features],
        "adversarial_y": y_adv,
    }


def _extract_input_feature_importance(fitted_pipeline, input_features):
    """
    Aggregate pipeline feature importances back to original input columns.

    Works best when the pipeline preprocessing step supports
    `get_feature_names_out()`, such as ColumnTransformer.

    Parameters
    ----------
    fitted_pipeline : fitted sklearn Pipeline
    input_features : list[str]
        Original input columns used to fit the pipeline.

    Returns
    -------
    pd.Series
        Importance per original input feature.
    """
    final_estimator = fitted_pipeline.steps[-1][1]
    raw_importances = final_estimator.feature_importances_

    # Case 1: no preprocessing or preprocessing preserves feature count
    if len(raw_importances) == len(input_features):
        return pd.Series(raw_importances, index=input_features)

    # Case 2: preprocessing step exposes transformed feature names
    if len(fitted_pipeline.steps) < 2:
        raise ValueError(
            "Could not map feature importances back to original columns."
        )

    preprocessor = fitted_pipeline.steps[-2][1]

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError(
            "Preprocessor does not provide get_feature_names_out(), so transformed "
            "feature importances cannot be mapped back to original columns."
        )

    transformed_names = preprocessor.get_feature_names_out()

    if len(transformed_names) != len(raw_importances):
        raise ValueError(
            "Mismatch between transformed feature names and feature importances."
        )

    # Aggregate transformed columns back to input columns.
    # Handles common ColumnTransformer naming like:
    #   num__age
    #   cat__city_London
    #   remainder__foo
    aggregated = {col: 0.0 for col in input_features}

    for name, imp in zip(transformed_names, raw_importances):
        matched = None

        # Exact suffix match is usually safest for ColumnTransformer output names
        for col in input_features:
            if name == col or name.endswith(f"__{col}") or f"__{col}_" in name:
                matched = col
                break

        # Fallback: first original column that is a suffix/prefix token match
        if matched is None:
            for col in input_features:
                if name.endswith(col) or name.startswith(col + "_"):
                    matched = col
                    break

        if matched is not None:
            aggregated[matched] += float(imp)

    series = pd.Series(aggregated)

    if series.sum() == 0:
        raise ValueError(
            "Failed to map transformed feature importances back to original columns."
        )

    return series
