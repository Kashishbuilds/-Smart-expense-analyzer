# pages/2_📊_Dashboard.py
# Main dashboard — shows analytics from YOUR entered expenses only

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_all_expenses
from utils.charts import (
    category_pie_chart,
    monthly_bar_chart,
    category_bar_chart,
    line_chart_monthly,
)

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Expense Dashboard")
st.markdown("A full overview of **your** spending habits and trends.")
st.divider()

# ----------------------------------------------------------------
# Load YOUR data
# ----------------------------------------------------------------
df = get_all_expenses()

if df.empty:
    st.warning(
        "📭 **No expenses recorded yet.**\n\n"
        "Head to **➕ Add Expense** in the sidebar to start adding your expenses. "
        "Your charts and analytics will appear here automatically."
    )
    st.stop()

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"]   = pd.to_datetime(df["date"], errors="coerce")

# ----------------------------------------------------------------
# KPI metrics
# ----------------------------------------------------------------
total_spent      = df["amount"].sum()
avg_expense      = df["amount"].mean()
num_transactions = len(df)
fraud_count      = len(df[df["is_fraud"] == 1])

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Spent",         f"₹{total_spent:,.0f}")
col2.metric("🧾 Transactions",        num_transactions)
col3.metric("📐 Avg per Transaction", f"₹{avg_expense:,.0f}")
col4.metric("⚠️ Fraud Alerts",       fraud_count,
            delta=f"{fraud_count} flagged" if fraud_count > 0 else "None",
            delta_color="inverse")

st.divider()

# ----------------------------------------------------------------
# Charts
# ----------------------------------------------------------------
col_left, col_right = st.columns(2)
with col_left:
    st.plotly_chart(category_pie_chart(df), use_container_width=True)
with col_right:
    st.plotly_chart(monthly_bar_chart(df), use_container_width=True)

col_left2, col_right2 = st.columns(2)
with col_left2:
    st.plotly_chart(category_bar_chart(df), use_container_width=True)
with col_right2:
    st.plotly_chart(line_chart_monthly(df), use_container_width=True)

st.divider()

# ----------------------------------------------------------------
# Filter & Browse
# ----------------------------------------------------------------
st.subheader("🔍 Filter & Browse Your Expenses")

categories   = ["All"] + sorted(df["category"].dropna().unique().tolist())
selected_cat = st.selectbox("Filter by Category:", categories)

filtered_df = df if selected_cat == "All" else df[df["category"] == selected_cat]
filtered_df = filtered_df.sort_values("date", ascending=False).copy()

filtered_df["date"]     = filtered_df["date"].dt.strftime("%Y-%m-%d")
filtered_df["amount"]   = filtered_df["amount"].apply(lambda x: f"₹{x:,.2f}")
filtered_df["is_fraud"] = filtered_df["is_fraud"].apply(lambda x: "⚠️ Flagged" if x == 1 else "✅ Normal")

filtered_df = filtered_df.rename(columns={
    "id": "ID", "amount": "Amount", "description": "Description",
    "category": "Category", "date": "Date", "is_fraud": "Status"
})

st.dataframe(
    filtered_df[["Date", "Description", "Amount", "Category", "Status"]],
    use_container_width=True, hide_index=True
)
st.caption(f"Showing {len(filtered_df)} expense(s) from your records")
