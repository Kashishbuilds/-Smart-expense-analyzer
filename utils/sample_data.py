# utils/sample_data.py
# Generates and loads sample expense data into the database for demo purposes

import random
from datetime import datetime, timedelta
from utils.database import add_expense, get_all_expenses

# Predefined sample expenses — realistic Indian spending patterns
SAMPLE_EXPENSES = [
    # Food
    (150, "Zomato biryani order", "Food"),
    (80, "Coffee at CCD", "Food"),
    (450, "Weekly groceries", "Food"),
    (200, "Lunch at office canteen", "Food"),
    (350, "Dinner with friends", "Food"),
    (60, "Tea and snacks", "Food"),
    (550, "Vegetable and fruit shopping", "Food"),
    (120, "Ice cream dessert", "Food"),
    (90, "Breakfast at hotel", "Food"),
    (280, "Swiggy pizza order", "Food"),

    # Travel
    (120, "Uber cab to office", "Travel"),
    (250, "Bus ticket Ahmedabad-Surat", "Travel"),
    (85, "Rapido bike ride", "Travel"),
    (500, "Petrol fill up car", "Travel"),
    (1200, "Train ticket to Mumbai", "Travel"),
    (70, "Metro card recharge", "Travel"),
    (180, "Auto rickshaw fare", "Travel"),
    (350, "Ola cab airport", "Travel"),

    # Bills
    (850, "Electricity bill payment", "Bills"),
    (599, "Airtel broadband monthly", "Bills"),
    (399, "Jio mobile recharge", "Bills"),
    (1500, "Gas cylinder booking", "Bills"),
    (149, "Netflix subscription", "Bills"),
    (119, "Spotify premium", "Bills"),
    (2500, "Home rent payment", "Bills"),
    (750, "Water tanker bill", "Bills"),

    # Shopping
    (1299, "Amazon t-shirt order", "Shopping"),
    (2500, "Flipkart headphones", "Shopping"),
    (890, "Shoes from mall", "Shopping"),
    (450, "Books from store", "Shopping"),
    (3500, "Mobile phone cover accessories", "Shopping"),
    (600, "Cosmetics from Nykaa", "Shopping"),
    (200, "Stationery for college", "Shopping"),
    (1800, "Jeans from Myntra", "Shopping"),
]


def load_sample_data():
    """
    Insert sample expenses into the database if it's nearly empty.
    Spreads data across the last 4 months for realistic trends.
    Only loads if fewer than 5 expenses exist (avoids duplicate loading).
    """
    existing = get_all_expenses()
    if len(existing) >= 5:
        return  # Data already exists, don't duplicate

    today = datetime.today()

    for amount, description, category in SAMPLE_EXPENSES:
        # Spread expenses randomly over last 4 months
        days_back = random.randint(0, 120)
        date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Small random variation in amounts to make data realistic
        varied_amount = round(amount * random.uniform(0.85, 1.15), 2)

        add_expense(
            amount=varied_amount,
            description=description,
            category=category,
            date=date,
            is_fraud=0
        )

    # Add a couple of "fraudulent" high-value anomaly transactions
    add_expense(
        amount=15000,
        description="Suspicious large online transfer",
        category="Shopping",
        date=today.strftime("%Y-%m-%d"),
        is_fraud=1
    )
    add_expense(
        amount=12500,
        description="Unusual international transaction",
        category="Shopping",
        date=(today - timedelta(days=5)).strftime("%Y-%m-%d"),
        is_fraud=1
    )
