# app.py
# ============================================================
# AI Smart Expense Analyzer
# Main entry point for the Streamlit application
#
# Run with:  streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import sys
import os

# ---- Ensure sub-modules are importable ----
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---- Initialize database and ML model on startup ----
from utils.database import init_db, get_all_expenses, get_monthly_spending, get_budget
from models.categorizer import train_model

# ------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Expense Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "# AI Smart Expense Analyzer\nBuilt with ❤️ using Streamlit + Scikit-learn",
    }
)

# ------------------------------------------------------------------
# One-time startup tasks (run only once per session)
# NOTE: Sample data loading removed — all data comes from user input
# ------------------------------------------------------------------
if "initialized" not in st.session_state:
    with st.spinner("🚀 Starting up..."):
        init_db()       # Create database tables if needed
        train_model()   # Train/cache the ML categorization model
    st.session_state.initialized = True

# ------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
    }

    /* Dividers */
    hr {
        border-color: #e2e8f0;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Dashboard (Home Page)
# ------------------------------------------------------------------
st.title("💰 AI Smart Expense Analyzer")
st.markdown(
    "Your intelligent personal finance companion powered by **Machine Learning**. "
    "Add your expenses manually and let AI categorize, predict, and protect your finances."
)
st.divider()

# ------------------------------------------------------------------
# Load real data
# ------------------------------------------------------------------
df = get_all_expenses()

if df.empty:
    # Welcome state — no data yet
    st.info(
        "👋 **Welcome!** You haven't added any expenses yet.\n\n"
        "➡️ Head to **➕ Add Expense** from the sidebar to get started. "
        "Once you add expenses, your dashboard, predictions, fraud detection, and AI chatbot will all come to life!"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧾 Total Expenses", "₹0.00")
    with col2:
        st.metric("📋 Transactions", "0")
    with col3:
        st.metric("⚠️ Flagged", "0")

    st.divider()
    st.subheader("🚀 How it works")
    st.markdown("""
    1. **➕ Add Expense** — Enter your spending details. AI will auto-categorize them.
    2. **📊 Dashboard** — See your spending breakdown by category and time.
    3. **📅 Budget Planner** — Set monthly budgets and track your usage.
    4. **🔮 Prediction** — AI predicts your next month's spending (needs 2+ months of data).
    5. **⚠️ Fraud Detection** — Isolation Forest flags unusual transactions automatically.
    6. **🤖 AI Chatbot** — Ask anything about your finances in natural language.
    7. **🗂️ Manage Expenses** — Filter, search, and delete your records.
    """)
    st.stop()

# ------------------------------------------------------------------
# Summary metrics
# ------------------------------------------------------------------
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"]   = pd.to_datetime(df["date"], errors="coerce")
df["month"]  = df["date"].dt.to_period("M").astype(str)

from datetime import datetime
cur_month   = datetime.today().strftime("%Y-%m")
month_df    = df[df["month"] == cur_month]
month_spent = month_df["amount"].sum()
budget      = get_budget(cur_month)
fraud_count = len(df[df["is_fraud"] == 1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("💸 Total All Time",  f"₹{df['amount'].sum():,.2f}")
col2.metric("📅 This Month",      f"₹{month_spent:,.2f}")
col3.metric("📋 Transactions",    len(df))
col4.metric("⚠️ Flagged",        fraud_count)

# ------------------------------------------------------------------
# Budget status
# ------------------------------------------------------------------
if budget > 0:
    pct = (month_spent / budget) * 100
    st.divider()
    st.subheader("🎯 Budget Status — This Month")
    if pct >= 100:
        st.error(f"🚨 Budget exceeded! Spent ₹{month_spent:,.2f} / ₹{budget:,.2f} ({pct:.1f}%)")
    elif pct >= 80:
        st.warning(f"⚠️ 80% budget used. Spent ₹{month_spent:,.2f} / ₹{budget:,.2f} ({pct:.1f}%)")
    else:
        st.success(f"✅ On track! Spent ₹{month_spent:,.2f} / ₹{budget:,.2f} ({pct:.1f}%)")

# ------------------------------------------------------------------
# Recent expenses
# ------------------------------------------------------------------
st.divider()
st.subheader("📋 Recent Expenses")
recent = df.head(8).copy()
recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
recent["Status"] = recent["is_fraud"].apply(lambda x: "⚠️ Flagged" if x == 1 else "✅ Normal")
recent = recent.rename(columns={
    "amount": "Amount (₹)", "description": "Description",
    "category": "Category", "date": "Date"
})
st.dataframe(recent[["Date","Description","Amount (₹)","Category","Status"]], use_container_width=True, hide_index=True)
