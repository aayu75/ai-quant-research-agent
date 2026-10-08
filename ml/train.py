import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from ml.features import create_features


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "ma_5",
    "ma_20",
    "ma_ratio",
    "volume_change"
]


def prepare_ml_data(
    data: pd.DataFrame
):

    df = create_features(data)

    if df.empty:

        raise ValueError(
            "Feature engineering produced "
            "no usable observations."
        )

    X = df[FEATURE_COLUMNS]

    y = df["target"]

    return X, y


def chronological_split(
    X,
    y,
    train_ratio=0.8
):

    if not 0 < train_ratio < 1:

        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if len(X) < 10:

        raise ValueError(
            "Not enough observations "
            "for chronological split."
        )

    split_index = int(
        len(X) * train_ratio
    )

    if split_index <= 0:

        raise ValueError(
            "Training split contains "
            "no observations."
        )

    if split_index >= len(X):

        raise ValueError(
            "Testing split contains "
            "no observations."
        )

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def validate_training_target(
    y_train
):

    unique_classes = (
        y_train
        .dropna()
        .unique()
    )

    if len(unique_classes) < 2:

        raise ValueError(
            "Training data contains only "
            "one target class. At least two "
            "classes are required."
        )


def build_logistic_model():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])


def build_random_forest_model():

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def train_model(
    data: pd.DataFrame
):

    X, y = prepare_ml_data(
        data
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = chronological_split(
        X,
        y
    )

    validate_training_target(
        y_train
    )

    model = build_logistic_model()

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


def train_multiple_models(
    data: pd.DataFrame
):

    X, y = prepare_ml_data(
        data
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = chronological_split(
        X,
        y
    )

    validate_training_target(
        y_train
    )

    models = {

        "logistic_regression":
            build_logistic_model(),

        "random_forest":
            build_random_forest_model()
    }

    results = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        results[name] = {

            "model": model,

            "X_train": X_train,

            "X_test": X_test,

            "y_train": y_train,

            "y_test": y_test
        }

    return results