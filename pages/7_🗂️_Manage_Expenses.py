# pages/7_🗂️_Manage_Expenses.py
# View, filter, edit, and delete your own expense records

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ IMPORT FIX (added clear_expenses)
from utils.database import get_all_expenses, delete_expense, clear_expenses

st.set_page_config(page_title="Manage Expenses", page_icon="🗂️", layout="wide")

st.title("🗂️ Manage Expenses")
st.markdown("View, filter, search, and delete **your own** expense records.")
st.divider()

# ----------------------------------------------------------------
# Load YOUR data
# ----------------------------------------------------------------
df = get_all_expenses()

if df.empty:
    st.info(
        "📭 **No expenses found yet.**\n\n"
        "Go to **➕ Add Expense** from the sidebar to start recording your spending."
    )
    st.stop()

df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["date"]   = pd.to_datetime(df["date"], errors="coerce")

# ----------------------------------------------------------------
# Filters
# ----------------------------------------------------------------
st.subheader("🔍 Filter Expenses")

col1, col2, col3 = st.columns(3)

with col1:
    categories   = ["All"] + sorted(df["category"].dropna().unique().tolist())
    selected_cat = st.selectbox("Category", categories)

with col2:
    min_date   = df["date"].min().date()
    max_date   = df["date"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

with col3:
    search_text = st.text_input("Search Description", placeholder="e.g. uber, pizza...")

# Apply filters
filtered = df.copy()

if selected_cat != "All":
    filtered = filtered[filtered["category"] == selected_cat]

if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["date"].dt.date >= start) &
        (filtered["date"].dt.date <= end)
    ]

if search_text:
    filtered = filtered[
        filtered["description"].str.contains(search_text, case=False, na=False)
    ]

filtered = filtered.sort_values("date", ascending=False)

# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
st.divider()
col_a, col_b, col_c = st.columns(3)
col_a.metric("📋 Records",  len(filtered))
col_b.metric("💰 Total",    f"₹{filtered['amount'].sum():,.2f}")
col_c.metric(
    "📐 Average",
    f"₹{filtered['amount'].mean():,.2f}" if len(filtered) > 0 else "₹0.00"
)

# ----------------------------------------------------------------
# Display table with delete buttons
# ----------------------------------------------------------------
st.divider()
st.subheader("📋 Your Expense Records")

if filtered.empty:
    st.info("No expenses match your filters.")
else:
    for _, row in filtered.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([2, 3, 2, 2, 1, 1])
        col1.write(row["date"].strftime("%Y-%m-%d"))
        col2.write(str(row["description"])[:45])
        col3.write(f"₹{row['amount']:,.2f}")
        col4.write(row["category"])
        col5.write("⚠️" if row["is_fraud"] == 1 else "✅")

        if col6.button("🗑️", key=f"del_{row['id']}", help="Delete this expense"):
            delete_expense(row["id"])
            st.success(f"Deleted: {row['description']}")
            st.rerun()

# ----------------------------------------------------------------
# Bulk clear (with ID reset FIX)
# ----------------------------------------------------------------
st.divider()
st.subheader("🗑️ Clear All Data")
st.markdown("**Warning:** This will permanently delete ALL your expense records.")

confirm_clear = st.checkbox("I understand this cannot be undone")

if st.button("🗑️ Delete All Expenses", type="secondary", disabled=not confirm_clear):
    # ✅ FIX: use clear_expenses() instead of raw SQL
    clear_expenses()
    st.success("✅ All expenses deleted and ID reset!")
    st.rerun()

# ----------------------------------------------------------------
# Export to CSV
# ----------------------------------------------------------------
st.divider()
st.subheader("📥 Export Data")

csv_data         = filtered.copy()
csv_data["date"] = csv_data["date"].dt.strftime("%Y-%m-%d")
csv_string       = csv_data.to_csv(index=False)

st.download_button(
    label="⬇️ Download as CSV",
    data=csv_string,
    file_name="my_expenses.csv",
    mime="text/csv",
    type="primary"
)