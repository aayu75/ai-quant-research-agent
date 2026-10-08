import pandas as pd


def prepare_price_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if hasattr(df.columns, "levels"):
        df.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in df.columns
        ]

    df = df.sort_index()

    required_columns = ["Close", "Volume"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    return df


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_price_data(data)

    # -----------------------------
    # Return-based features
    # -----------------------------

    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_20d"] = df["Close"].pct_change(20)

    # -----------------------------
    # Volatility
    # -----------------------------

    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(window=20)
        .std()
    )

    # -----------------------------
    # Moving averages
    # -----------------------------

    df["ma_5"] = (
        df["Close"]
        .rolling(window=5)
        .mean()
    )

    df["ma_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["ma_ratio"] = (
        df["ma_5"] /
        df["ma_20"]
    )

    # -----------------------------
    # Volume feature
    # -----------------------------

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    # -----------------------------
    # Target
    # -----------------------------

    # Predict whether the NEXT day's
    # return will be positive.

    df["next_day_return"] = (
        df["Close"].shift(-1)
        / df["Close"]
        - 1
    )

    feature_columns = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "ma_5",
        "ma_20",
        "ma_ratio",
        "volume_change"
    ]

    # Remove rows where either the
    # features or true future return
    # is unavailable.
    df = df.dropna(
        subset=feature_columns + ["next_day_return"]
    )

    df["target"] = (
        df["next_day_return"] > 0
    ).astype(int)

    df = df.drop(
        columns=["next_day_return"]
    )

    return df