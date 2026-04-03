# pages/6_🤖_AI_Chatbot.py
# ============================================================
# AI Finance Chatbot — Streamlit Page
# Fetches LIVE data from SQLite on every message.
# Every response is dynamically computed from real expenses.
# ============================================================

import streamlit as st
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import get_all_expenses, get_budget
from utils.chatbot import get_chatbot_response

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* Quick-question chip buttons */
div[data-testid="stHorizontalBlock"] button {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    padding: 4px 14px !important;
    color: #334155 !important;
    transition: all 0.15s;
}
div[data-testid="stHorizontalBlock"] button:hover {
    background: #e0e7ff !important;
    border-color: #6366f1 !important;
    color: #4338ca !important;
}

/* Live stats bar */
.stat-chip {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 2px 4px;
}

/* Chat message box */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 4px 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────
st.title("🤖 AI Finance Chatbot")
st.markdown(
    "I analyze **your personally entered expenses** from the database and compute every answer "
    "dynamically — no hardcoded responses."
)

# ── Live data bar ─────────────────────────────────────────────
df = get_all_expenses()

if not df.empty:
    import pandas as pd
    df_disp = df.copy()
    df_disp["amount"] = pd.to_numeric(df_disp["amount"], errors="coerce").fillna(0)
    df_disp["date"]   = pd.to_datetime(df_disp["date"], errors="coerce")
    df_disp["month"]  = df_disp["date"].dt.to_period("M").astype(str)

    now           = datetime.now()
    cur_month     = now.strftime("%Y-%m")
    total         = df_disp["amount"].sum()
    month_spent   = df_disp[df_disp["month"] == cur_month]["amount"].sum()
    txn_count     = len(df_disp)
    fraud_count   = len(df_disp[df_disp["is_fraud"] == 1])
    budget        = get_budget(cur_month)
    budget_pct    = f"{(month_spent/budget*100):.0f}% budget used" if budget > 0 else "No budget set"

    st.markdown(
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:10px 16px;margin-bottom:12px">'
        f'📡 <b>Live DB Snapshot</b> &nbsp;|&nbsp; '
        f'<span class="stat-chip">₹{total:,.0f} total</span>'
        f'<span class="stat-chip">₹{month_spent:,.0f} this month</span>'
        f'<span class="stat-chip">{txn_count} transactions</span>'
        f'<span class="stat-chip">{budget_pct}</span>'
        + (f'<span class="stat-chip" style="background:linear-gradient(135deg,#ef4444,#f97316)">⚠️ {fraud_count} flagged</span>' if fraud_count > 0 else '') +
        f'</div>',
        unsafe_allow_html=True
    )

st.divider()

# ── Initialize chat history in session state ──────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

    # Build a live welcome message based on real data
    if not df.empty:
        welcome = get_chatbot_response("hello", df, budget_fn=get_budget)
    else:
        welcome = (
            "👋 Hello! I'm your **AI Finance Assistant**.\n\n"
            "No expenses found yet — head to **Add Expense** to start logging, "
            "then come back and ask me anything!"
        )
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": welcome
    })

# ── Quick-question chips ──────────────────────────────────────
st.markdown("**💬 Quick Questions** — click to ask instantly:")

# Organised by topic
QUICK_QUESTIONS = {
    "📊 Spending": [
        "What's my total spending?",
        "Show this month's spending",
        "Show last month's summary",
    ],
    "🏆 Categories": [
        "Where did I spend the most?",
        "Show category breakdown",
        "Compare food vs travel",
    ],
    "🔎 Transactions": [
        "Show my recent expenses",
        "What's my largest transaction?",
        "What are my most frequent expenses?",
    ],
    "📅 Time Analysis": [
        "What did I spend this week?",
        "What's my spending trend?",
        "What was my worst month?",
        "What's my daily average?",
    ],
    "⚠️ Alerts": [
        "Are there any fraud alerts?",
        "What's my budget status?",
        "Predict next month's spending",
    ],
    "💡 Advice": [
        "How can I save money?",
        "What percentage goes to shopping?",
        "Which day do I spend the most?",
    ],
}

for section, questions in QUICK_QUESTIONS.items():
    cols = st.columns(len(questions))
    # Show section label in first column only
    for i, (col, q) in enumerate(zip(cols, questions)):
        if col.button(q, key=f"quick_{q}", use_container_width=True):
            # Fetch fresh data at click time
            live_df = get_all_expenses()
            st.session_state.chat_history.append({"role": "user", "content": q})
            response = get_chatbot_response(q, live_df, budget_fn=get_budget)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

st.divider()

# ── Chat history display ──────────────────────────────────────
st.subheader(f"💬 Conversation  ·  {len(st.session_state.chat_history)} messages")

for msg in st.session_state.chat_history:
    is_bot = msg["role"] == "assistant"
    with st.chat_message(msg["role"], avatar="🤖" if is_bot else "👤"):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask me anything… e.g. 'How much on food?', 'Compare bills vs shopping', 'Best month?'"
)

if user_input:
    # Always fetch fresh data from SQLite when user sends a message
    live_df = get_all_expenses()

    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate response with real live data
    response = get_chatbot_response(user_input, live_df, budget_fn=get_budget)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.rerun()

# ── Bottom controls ───────────────────────────────────────────
st.divider()
col_clear, col_info = st.columns([1, 3])

with col_clear:
    if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Chat cleared! Ask me anything about your finances 😊"
        }]
        st.rerun()

with col_info:
    st.caption(
        "🔄 Every response is computed live from your SQLite database. "
        "Add new expenses and ask again — answers update instantly!"
    )

# ── Sidebar: what I can do ────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Chatbot Capabilities")
    st.markdown("""
    **Real-time analysis of your data:**

    📊 Total & monthly spending  
    🏆 Category ranking  
    📋 Recent transactions  
    🔺 Largest/smallest amounts  
    📅 Today / week / month  
    📈 Spending trends  
    🌟 Best & worst months  
    ⚖️ Category comparisons  
    💡 Personalized savings tips  
    ⚠️ Fraud alerts  
    📅 Budget status  
    🔮 Next month prediction  
    📐 Averages & daily spend  
    🏖️ Weekend vs weekday  
    📅 Day-of-week patterns  
    🔄 Most frequent expenses  

    ---
    _All data pulled live from SQLite._
    """)
