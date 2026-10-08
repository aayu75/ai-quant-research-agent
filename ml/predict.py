import pandas as pd

from ml.features import create_features

from ml.train import (
    FEATURE_COLUMNS,
    build_logistic_model,
    build_random_forest_model,
    prepare_ml_data,
    chronological_split
)


def train_prediction_model(
    data: pd.DataFrame,
    model_name: str = "logistic_regression"
):
    """
    Train a prediction model using the chronological
    train/test split.

    This function is useful when we want to evaluate
    a model before making a prediction.
    """

    X, y = prepare_ml_data(data)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = chronological_split(
        X,
        y
    )

    if model_name == "logistic_regression":

        model = build_logistic_model()

    elif model_name == "random_forest":

        model = build_random_forest_model()

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model.fit(
        X_train,
        y_train
    )

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def retrain_model_on_all_data(
    data: pd.DataFrame,
    model_name: str
):
    """
    Retrain the selected model using ALL available
    labeled historical observations.

    This should only be done AFTER model evaluation
    and model selection.

    The resulting model is used for the final
    next-day prediction.
    """

    X, y = prepare_ml_data(data)

    if model_name == "logistic_regression":

        model = build_logistic_model()

    elif model_name == "random_forest":

        model = build_random_forest_model()

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model.fit(
        X,
        y
    )

    return {
        "model": model,
        "X": X,
        "y": y,
        "observations": len(X)
    }


def predict_test_period(
    model,
    X_test
):
    """
    Generate predictions for the held-out
    test period.
    """

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    return (
        predictions,
        probabilities
    )


def predict_next_day(
    model,
    data: pd.DataFrame
):
    """
    Generate a next-day directional prediction.

    The model supplied here should be the model
    retrained on all available historical data.
    """

    features = create_features(
        data
    )

    if features.empty:

        raise ValueError(
            "No usable features available "
            "for next-day prediction."
        )

    latest_features = (
        features[
            FEATURE_COLUMNS
        ].iloc[[-1]]
    )

    prediction = int(
        model.predict(
            latest_features
        )[0]
    )

    probabilities = model.predict_proba(
        latest_features
    )[0]

    probability = float(
        probabilities[prediction]
    )

    return {
        "prediction": prediction,
        "probability": probability
    }