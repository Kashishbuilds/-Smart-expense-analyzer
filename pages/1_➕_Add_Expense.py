# pages/1_➕_Add_Expense.py
# Page for entering new expenses with ML categorization and fraud detection

import streamlit as st
import pandas as pd
from datetime import date
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import add_expense, get_all_expenses
from models.categorizer import predict_category
from models.fraud_detector import detect_fraud

st.set_page_config(page_title="Add Expense", page_icon="➕", layout="wide")

st.title("➕ Add New Expense")
st.markdown(
    "Enter your expense details below. "
    "AI will **automatically categorize** it and **check for fraud** based on your spending history."
)
st.divider()

# ----------------------------------------------------------------
# Expense entry form
# ----------------------------------------------------------------
with st.form("expense_form", clear_on_submit=True):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "💰 Amount (₹)",
            min_value=0.01,
            max_value=1_000_000.0,
            value=None,
            step=1.0,
            format="%.2f",
            placeholder="e.g. 350.00",
            help="Enter the expense amount in rupees"
        )
        description = st.text_input(
            "📝 Description",
            placeholder="e.g. Uber cab to office, Pizza delivery, Electricity bill...",
            help="Describe what you spent on — AI will use this to classify the category"
        )

    with col2:
        expense_date = st.date_input(
            "📅 Date",
            value=date.today(),
            help="When did this expense happen?"
        )
        manual_category = st.selectbox(
            "🏷️ Category",
            options=["Auto-detect (AI)", "Food", "Travel", "Bills", "Shopping"],
            help="Leave as 'Auto-detect' to let AI classify it, or pick manually"
        )

    notes = st.text_area(
        "🗒️ Notes (optional)",
        placeholder="Any extra details about this expense...",
        height=68
    )

    submit_button = st.form_submit_button("💾 Save Expense", use_container_width=True, type="primary")

# ----------------------------------------------------------------
# Handle form submission
# ----------------------------------------------------------------
if submit_button:
    if amount is None or amount <= 0:
        st.error("⚠️ Please enter a valid amount greater than ₹0.")
    elif not description.strip():
        st.error("⚠️ Please enter a description for the expense.")
    else:
        # Step 1: Categorize using ML (or use manual override)
        if manual_category == "Auto-detect (AI)":
            category = predict_category(description)
            st.info(f"🤖 AI detected category: **{category}**")
        else:
            category = manual_category

        # Step 2: Fraud detection using Isolation Forest on YOUR data
        all_expenses = get_all_expenses()
        past_amounts = all_expenses["amount"].tolist() if not all_expenses.empty else []
        is_fraud = detect_fraud(past_amounts, amount)

        # Step 3: Save to database
        full_description = description.strip()
        if notes.strip():
            full_description += f" | {notes.strip()}"

        add_expense(
            amount=amount,
            description=full_description,
            category=category,
            date=str(expense_date),
            is_fraud=int(is_fraud)
        )

        # Step 4: Show result
        if is_fraud:
            st.warning(
                f"⚠️ **Fraud Alert!** This expense of ₹{amount:,.2f} looks **unusual** "
                f"compared to your spending history. It has been saved and flagged for review."
            )
        else:
            st.success(
                f"✅ Expense saved! **₹{amount:,.2f}** for *{description}* under **{category}**."
            )
        st.balloons()

# ----------------------------------------------------------------
# Helper tip
# ----------------------------------------------------------------
all_df = get_all_expenses()
if all_df.empty:
    st.divider()
    st.info(
        "💡 **Tip:** Start adding your expenses here. "
        "Once you have data, head to **Dashboard**, **Budget Planner**, **Prediction**, or the **AI Chatbot** to explore insights."
    )
else:
    st.divider()
    st.subheader("📋 Your Recent Expenses")
    display_df = all_df.head(10).copy()
    display_df["is_fraud"] = display_df["is_fraud"].apply(lambda x: "⚠️ Flagged" if x == 1 else "✅ Normal")
    display_df = display_df.rename(columns={
        "id": "ID", "amount": "Amount (₹)", "description": "Description",
        "category": "Category", "date": "Date", "is_fraud": "Status"
    })
    st.dataframe(display_df[["ID","Date","Description","Amount (₹)","Category","Status"]], use_container_width=True, hide_index=True)
