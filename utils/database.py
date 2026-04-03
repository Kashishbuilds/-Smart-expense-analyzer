# utils/database.py
# Handles all SQLite database operations

import sqlite3
import pandas as pd
from datetime import datetime

# Path to the SQLite database file
DB_PATH = "database.db"


def get_connection():
    """Create and return a database connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    """
    Initialize the database.
    Creates tables if they don't exist yet.
    Called once when the app starts.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Create the expenses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            amount      REAL    NOT NULL,
            description TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            is_fraud    INTEGER DEFAULT 0
        )
    """)

    # Create the budget table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS budget (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            month      TEXT    NOT NULL UNIQUE,
            amount     REAL    NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def add_expense(amount: float, description: str, category: str, date: str, is_fraud: int = 0):
    """Insert a new expense record into the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO expenses (amount, description, category, date, is_fraud) VALUES (?, ?, ?, ?, ?)",
        (amount, description, category, date, is_fraud)
    )

    conn.commit()
    conn.close()


def get_all_expenses() -> pd.DataFrame:
    """Fetch all expenses and return as a Pandas DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM expenses ORDER BY date DESC", conn)
    conn.close()
    return df


def delete_expense(expense_id: int):
    """Delete an expense by its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))

    conn.commit()
    conn.close()


# ✅ FINAL FIX (STRONG VERSION)
def clear_expenses():
    """Delete all expenses AND reset ID counter to 1 safely."""
    conn = get_connection()
    cursor = conn.cursor()

    # Delete all rows
    cursor.execute("DELETE FROM expenses")

    # Reset auto-increment ID (only if table exists in sequence)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'")
    if cursor.fetchone():
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='expenses'")

    conn.commit()
    conn.close()


def set_budget(month: str, amount: float):
    """Insert or update the budget for a given month."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO budget (month, amount) VALUES (?, ?) "
        "ON CONFLICT(month) DO UPDATE SET amount=excluded.amount",
        (month, amount)
    )

    conn.commit()
    conn.close()


def get_budget(month: str) -> float:
    """Return the budget for a given month, or 0.0 if not set."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT amount FROM budget WHERE month = ?", (month,))
    row = cursor.fetchone()

    conn.close()
    return row[0] if row else 0.0


def get_monthly_spending(month: str) -> float:
    """Return total spending for a given month (format 'YYYY-MM')."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT SUM(amount) FROM expenses WHERE strftime('%Y-%m', date) = ?",
        (month,)
    )

    row = cursor.fetchone()
    conn.close()
    return row[0] if row[0] else 0.0