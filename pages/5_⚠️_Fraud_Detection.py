# pages/5_⚠️_Fraud_Detection.py
# Displays detected anomalies using Isolation Forest on user's own data

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_all_expenses, get_connection
from models.fraud_detector import detect_fraud, get_fraud_score

st.set_page_config(page_title="Fraud Detection", page_icon="⚠️", layout="wide")

st.title("⚠️ Fraud & Anomaly Detection")
st.markdown(
    "This module uses **Isolation Forest** — an unsupervised ML algorithm — "
    "to detect unusually large or suspicious transactions in **your** expense history."
)
st.divider()

# ----------------------------------------------------------------
# Load YOUR data
# ----------------------------------------------------------------
df = get_all_expenses()

if df.empty:
    st.warning(
        "📭 **No expense data yet.**\n\n"
        "Add expenses via **➕ Add Expense**. "
        "Once you have at least **5 transactions**, the fraud detection algorithm will start flagging anomalies."
    )
    st.stop()

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"]   = pd.to_datetime(df["date"], errors="coerce")

if len(df) < 5:
    remaining = 5 - len(df)
    st.info(
        f"📊 Fraud detection needs at least **5 transactions** to work. "
        f"You have **{len(df)}** — add **{remaining} more** to enable it.\n\n"
        f"Every new expense added on the **➕ Add Expense** page is automatically checked."
    )
    # Show existing data
    st.subheader("Your Expenses So Far")
    display = df.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display[["date","description","amount","category"]], use_container_width=True, hide_index=True)
    st.stop()

# ----------------------------------------------------------------
# Summary metrics
# ----------------------------------------------------------------
total   = len(df)
flagged = len(df[df["is_fraud"] == 1])
normal  = total - flagged

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🧾 Total Transactions", total)
with col2:
    st.metric("✅ Normal", normal)
with col3:
    st.metric("⚠️ Suspicious", flagged, delta_color="inverse")

# ----------------------------------------------------------------
# Fraud alert section
# ----------------------------------------------------------------
st.divider()
if flagged > 0:
    st.error(f"🚨 **{flagged} suspicious transaction(s) detected!** Please review them carefully.")
    fraud_df = df[df["is_fraud"] == 1].copy()
    fraud_df["date"] = fraud_df["date"].dt.strftime("%Y-%m-%d")

    for _, row in fraud_df.iterrows():
        with st.container():
            col_a, col_b, col_c, col_d = st.columns([2, 3, 2, 2])
            col_a.markdown(f"**📅 {row['date']}**")
            col_b.markdown(f"📝 {row['description']}")
            col_c.markdown(f"💰 ₹{row['amount']:,.2f}")
            col_d.markdown(f"🏷️ {row['category']}")
        st.divider()
else:
    st.success("✅ **No suspicious transactions found.** Your spending looks normal!")

# ----------------------------------------------------------------
# Re-scan button
# ----------------------------------------------------------------
st.subheader("🔄 Re-scan All Expenses")
st.markdown(
    "Click below to re-run the fraud detection algorithm on all your recorded expenses. "
    "Useful after adding many new transactions."
)

if st.button("🔍 Run Fraud Detection Scan", type="primary"):
    with st.spinner("Scanning all your expenses..."):
        amounts = df["amount"].tolist()
        conn    = get_connection()
        cursor  = conn.cursor()

        updated = 0
        for idx, row in df.iterrows():
            other_amounts = [a for i, a in enumerate(amounts) if i != idx]
            is_fraud = detect_fraud(other_amounts, row["amount"])
            cursor.execute(
                "UPDATE expenses SET is_fraud = ? WHERE id = ?",
                (int(is_fraud), row["id"])
            )
            if is_fraud:
                updated += 1

        conn.commit()
        conn.close()

    st.success(f"✅ Scan complete! Found **{updated}** suspicious transaction(s).")
    st.rerun()

# ----------------------------------------------------------------
# Scatter plot
# ----------------------------------------------------------------
st.divider()
st.subheader("📈 Expense Amount Distribution")

plot_df = df.copy()
plot_df["date_str"] = plot_df["date"].dt.strftime("%Y-%m-%d")
plot_df["Fraud"]    = plot_df["is_fraud"].apply(lambda x: "⚠️ Suspicious" if x == 1 else "✅ Normal")

fig = px.scatter(
    plot_df,
    x="date_str",
    y="amount",
    color="Fraud",
    hover_data=["description", "category"],
    title="All Your Transactions — Normal vs Suspicious",
    color_discrete_map={"⚠️ Suspicious": "#ef4444", "✅ Normal": "#22c55e"},
    labels={"date_str": "Date", "amount": "Amount (₹)"},
)
fig.update_traces(marker_size=10)
fig.update_layout(title_font_size=16, xaxis_tickangle=-45, margin=dict(b=80))
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------
# All transactions table
# ----------------------------------------------------------------
st.divider()
st.subheader("📋 All Your Transactions")

display_df = plot_df.copy()
display_df["date"]   = display_df["date_str"]
display_df["amount"] = display_df["amount"].apply(lambda x: f"₹{x:,.2f}")
display_df = display_df.rename(columns={
    "date": "Date", "description": "Description",
    "amount": "Amount", "category": "Category", "Fraud": "Status"
})
st.dataframe(
    display_df[["Date", "Description", "Amount", "Category", "Status"]],
    use_container_width=True,
    hide_index=True
)
