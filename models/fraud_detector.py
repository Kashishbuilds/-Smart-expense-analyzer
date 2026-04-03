# models/fraud_detector.py
# Uses Isolation Forest to detect anomalous (unusually high) expenses

import numpy as np
from sklearn.ensemble import IsolationForest


def detect_fraud(amounts: list, new_amount: float) -> bool:
    """
    Determine if a new expense amount looks anomalous compared to past expenses.

    How Isolation Forest works:
    - It randomly partitions data into trees
    - Outliers (anomalies) are isolated in fewer splits
    - Points that are easy to isolate get a low anomaly score

    Parameters:
        amounts    - list of past expense amounts (at least 5 needed)
        new_amount - the new amount to evaluate

    Returns:
        True  → the amount looks suspicious / anomalous
        False → the amount looks normal
    """

    # We need enough historical data to make a meaningful comparison
    if len(amounts) < 5:
        return False  # Not enough data to judge

    # Reshape to 2D array as required by sklearn
    all_amounts = np.array(amounts + [new_amount]).reshape(-1, 1)

    # contamination = expected proportion of outliers in the data
    # 0.1 means we expect roughly 10% of expenses to be unusual
    model = IsolationForest(
        contamination=0.1,
        random_state=42,
        n_estimators=100
    )
    model.fit(all_amounts)

    # Predict: -1 = anomaly (fraud), 1 = normal
    prediction = model.predict([[new_amount]])
    return prediction[0] == -1


def get_fraud_score(amounts: list, new_amount: float) -> float:
    """
    Return an anomaly score for the new amount.
    Lower (more negative) score = more anomalous.
    """
    if len(amounts) < 5:
        return 0.0

    all_amounts = np.array(amounts + [new_amount]).reshape(-1, 1)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(all_amounts)

    score = model.decision_function([[new_amount]])
    return float(score[0])
