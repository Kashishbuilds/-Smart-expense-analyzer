# pages/3_📅_Budget_Planner.py
# Monthly budget tracker — works entirely on user-entered expenses

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import set_budget, get_budget, get_monthly_spending, get_all_expenses
from utils.charts import budget_gauge

st.set_page_config(page_title="Budget Planner", page_icon="📅", layout="wide")

st.title("📅 Budget Planner")
st.markdown("Set your monthly spending limits and track how you're doing against your **own expenses**.")
st.divider()

# ----------------------------------------------------------------
# Month selector
# ----------------------------------------------------------------
today         = datetime.today()
current_month = today.strftime("%Y-%m")

months = []
for i in range(-2, 2):
    m = (today.replace(day=1) + timedelta(days=32 * i)).strftime("%Y-%m")
    months.append(m)

selected_month = st.selectbox(
    "📆 Select Month:",
    options=months,
    index=months.index(current_month) if current_month in months else 0
)

# ----------------------------------------------------------------
# Budget setup
# ----------------------------------------------------------------
st.subheader(f"💰 Set Budget for {selected_month}")

current_budget = get_budget(selected_month)

col1, col2 = st.columns([2, 1])
with col1:
    new_budget = st.number_input(
        "Monthly Budget (₹)",
        min_value=0.0,
        value=float(current_budget) if current_budget > 0 else 10000.0,
        step=500.0,
        format="%.2f",
        help="Set how much you plan to spend this month"
    )
with col2:
    st.write("")
    st.write("")
    if st.button("💾 Save Budget", type="primary", use_container_width=True):
        set_budget(selected_month, new_budget)
        st.success(f"✅ Budget of ₹{new_budget:,.2f} saved for {selected_month}!")
        st.rerun()

# ----------------------------------------------------------------
# Spending vs Budget
# ----------------------------------------------------------------
st.divider()
st.subheader(f"📊 Spending Analysis: {selected_month}")

budget = get_budget(selected_month)
spent  = get_monthly_spending(selected_month)

if spent == 0 and budget == 0:
    st.info(
        "ℹ️ No expenses or budget set for this month yet.\n\n"
        "👉 Add expenses via **➕ Add Expense**, then set a budget above to start tracking!"
    )
elif budget <= 0:
    st.info(
        f"ℹ️ You've spent **₹{spent:,.2f}** this month but haven't set a budget yet. "
        f"Set one above to track your progress!"
    )
else:
    remaining = budget - spent
    pct_used  = (spent / budget) * 100

    if pct_used >= 100:
        st.error(
            f"🚨 **Budget Exceeded!** You've spent ₹{spent:,.2f} against a budget of ₹{budget:,.2f}. "
            f"You've gone over by ₹{abs(remaining):,.2f}!"
        )
    elif pct_used >= 80:
        st.warning(
            f"⚠️ **80% Warning!** You've used {pct_used:.1f}% of your budget. "
            f"Only ₹{remaining:,.2f} remaining!"
        )
    else:
        st.success(
            f"✅ You're doing great! {pct_used:.1f}% of budget used. "
            f"₹{remaining:,.2f} remaining."
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Budget",    f"₹{budget:,.2f}")
    with col2:
        st.metric("💸 Spent",    f"₹{spent:,.2f}", delta=f"{pct_used:.1f}% used")
    with col3:
        st.metric(
            "🏦 Remaining", f"₹{remaining:,.2f}",
            delta_color="normal" if remaining >= 0 else "inverse",
        )

    fig_gauge = budget_gauge(spent, budget)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------------------------------------------------------
# All months budget overview
# ----------------------------------------------------------------
st.divider()
st.subheader("📋 All Months Budget Summary")

df = get_all_expenses()
if not df.empty:
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly_spend = df.groupby("month")["amount"].sum().reset_index()
    monthly_spend.columns = ["Month", "Spent (₹)"]
    monthly_spend = monthly_spend.sort_values("Month", ascending=False)

    monthly_spend["Budget (₹)"]    = monthly_spend["Month"].apply(get_budget)
    monthly_spend["Remaining (₹)"] = monthly_spend["Budget (₹)"] - monthly_spend["Spent (₹)"]
    monthly_spend["% Used"]        = monthly_spend.apply(
        lambda r: f"{(r['Spent (₹)']/r['Budget (₹)']*100):.1f}%" if r["Budget (₹)"] > 0 else "No Budget",
        axis=1
    )

    for col in ["Spent (₹)", "Budget (₹)", "Remaining (₹)"]:
        monthly_spend[col] = monthly_spend[col].apply(lambda x: f"₹{x:,.2f}")

    st.dataframe(monthly_spend, use_container_width=True, hide_index=True)
else:
    st.info(
        "No expenses recorded yet. "
        "Head to **➕ Add Expense** to start tracking your spending!"
    )
