# pages/4_🔮_Prediction.py
# Predicts next month's total spending using Linear Regression
# Works entirely on user-entered data

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_all_expenses
from models.predictor import predict_next_month

st.set_page_config(page_title="Expense Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Expense Prediction")
st.markdown(
    "Using **Linear Regression**, this tool analyzes your past spending and predicts "
    "how much you'll likely spend next month."
)
st.divider()

# ----------------------------------------------------------------
# Load and prepare YOUR data
# ----------------------------------------------------------------
df = get_all_expenses()

if df.empty:
    st.warning(
        "📭 **No expense data yet.**\n\n"
        "Head to **➕ Add Expense** and start recording your expenses. "
        "Once you have data from at least **2 different months**, predictions will appear here."
    )
    st.stop()

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"]   = pd.to_datetime(df["date"], errors="coerce")
df["month"]  = df["date"].dt.to_period("M").astype(str)

# Group by month
monthly = df.groupby("month")["amount"].sum().reset_index()
monthly.columns = ["Month", "Total (₹)"]
monthly = monthly.sort_values("Month")

if len(monthly) < 2:
    months_needed = 2 - len(monthly)
    st.info(
        f"📊 **Need more data!** You have data from **{len(monthly)} month(s)**. "
        f"Please add expenses for **{months_needed} more month(s)** to enable predictions.\n\n"
        f"Keep adding expenses over time and come back!"
    )
    # Still show what we have
    if not monthly.empty:
        st.divider()
        st.subheader("📋 Your Monthly Data So Far")
        display = monthly.copy()
        display["Total (₹)"] = display["Total (₹)"].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
    st.stop()

# ----------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------
monthly_totals = monthly["Total (₹)"].tolist()
predicted = predict_next_month(monthly_totals)

st.subheader("📈 Next Month's Predicted Spending")

col1, col2, col3 = st.columns(3)
with col1:
    last_month_spend = monthly_totals[-1]
    st.metric("📅 Last Month", f"₹{last_month_spend:,.0f}")
with col2:
    st.metric(
        "🔮 Predicted (Next Month)",
        f"₹{predicted:,.0f}",
        delta=f"₹{predicted - last_month_spend:,.0f} vs last month",
        delta_color="inverse" if predicted > last_month_spend else "normal"
    )
with col3:
    avg = sum(monthly_totals) / len(monthly_totals)
    st.metric("📊 Historical Average", f"₹{avg:,.0f}")

# ---- Chart: actual + prediction ----
st.divider()
st.subheader("📊 Historical + Predicted Chart")

last_period = pd.Period(monthly["Month"].iloc[-1], freq="M")
next_period = str(last_period + 1)

months_all = monthly["Month"].tolist() + [next_period]

fig = go.Figure()

# Actual spending line
fig.add_trace(go.Scatter(
    x=monthly["Month"].tolist(),
    y=monthly["Total (₹)"].tolist(),
    mode="lines+markers",
    name="Your Actual Spending",
    line=dict(color="#6366f1", width=3),
    marker=dict(size=8),
))

# Predicted point (connected from last actual)
fig.add_trace(go.Scatter(
    x=[monthly["Month"].iloc[-1], next_period],
    y=[monthly["Total (₹)"].iloc[-1], predicted],
    mode="lines+markers",
    name="Predicted",
    line=dict(color="#f97316", width=3, dash="dash"),
    marker=dict(size=12, symbol="star"),
))

fig.update_layout(
    title="Your Monthly Spending + AI Prediction",
    xaxis_title="Month",
    yaxis_title="Total Spending (₹)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    title_font_size=18,
    margin=dict(t=60, b=60),
)
st.plotly_chart(fig, use_container_width=True)

# ---- Advice ----
st.divider()
st.subheader("💡 AI Savings Suggestion")

if predicted > last_month_spend * 1.1:
    st.warning(
        f"📈 Your spending is predicted to **increase by {((predicted/last_month_spend)-1)*100:.1f}%** next month. "
        f"Consider reviewing your budget and cutting discretionary expenses."
    )
elif predicted < last_month_spend * 0.9:
    st.success(
        f"📉 Great news! Your spending is predicted to **decrease by {((1-predicted/last_month_spend))*100:.1f}%** next month. "
        f"Keep up the good habits!"
    )
else:
    st.info(
        f"📊 Your spending looks stable. Predicted ₹{predicted:,.0f} vs last month's ₹{last_month_spend:,.0f}."
    )

# ----------------------------------------------------------------
# Monthly data table
# ----------------------------------------------------------------
st.divider()
st.subheader("📋 Your Monthly Spending Data")
display = monthly.copy()
display["Total (₹)"] = display["Total (₹)"].apply(lambda x: f"₹{x:,.2f}")
st.dataframe(display.sort_values("Month", ascending=False), use_container_width=True, hide_index=True)
