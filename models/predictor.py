# models/predictor.py
# Uses Linear Regression to predict next month's total spending

import numpy as np
from sklearn.linear_model import LinearRegression


def predict_next_month(monthly_totals: list) -> float:
    """
    Predict the next month's total expense using Linear Regression.

    The idea:
    - X = month numbers (1, 2, 3, ...)
    - Y = total spending for each month
    - We fit a straight line and extrapolate for the next month

    Parameters:
        monthly_totals - list of floats, one per month in chronological order
                         e.g. [3500.0, 4200.0, 3800.0, 5100.0]

    Returns:
        Predicted spending for the next month (float)
        Returns 0.0 if not enough data (need at least 2 months)
    """

    if len(monthly_totals) < 2:
        return 0.0

    n = len(monthly_totals)

    # X = month index: [[1], [2], [3], ...]
    X = np.arange(1, n + 1).reshape(-1, 1)
    # Y = actual spending for each month
    Y = np.array(monthly_totals)

    model = LinearRegression()
    model.fit(X, Y)

    # Predict for month n+1
    next_month_index = np.array([[n + 1]])
    prediction = model.predict(next_month_index)[0]

    # Spending can't be negative
    return max(0.0, round(prediction, 2))
