# utils/chatbot.py
# ============================================================
# AI Finance Chatbot — Real Data Engine
#
# Every single response is computed LIVE from SQLite database.
# No hardcoded answers. All numbers, trends, and advice are
# derived from the user's actual expense records.
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────
# INTENT DETECTION
# Maps user message keywords → intent labels
# A message can trigger multiple intents simultaneously
# ─────────────────────────────────────────────────────────────

INTENT_MAP = {
    "greeting":           ["hello", "hi", "hey", "namaste", "good morning", "good evening", "sup", "hiya", "start"],
    "help":               ["help", "what can you do", "commands", "options", "menu", "capabilities", "guide"],
    "total_spending":     ["total", "overall", "all time", "how much have i spent", "spent so far", "grand total"],
    "this_month":         ["this month", "current month", "month so far", "spending this month", "monthly spending"],
    "last_month":         ["last month", "previous month", "past month", "month before"],
    "top_category":       ["most", "highest", "where did i spend", "top category", "biggest expense", "max spend", "most money on"],
    "lowest_category":    ["least", "lowest", "minimum", "smallest category", "least money", "spend least"],
    "category_breakdown": ["breakdown", "split", "categories", "by category", "category wise", "each category", "all categories"],
    "food":               ["food", "eating", "restaurant", "groceries", "meals", "dining", "zomato", "swiggy", "lunch", "dinner", "breakfast"],
    "travel":             ["travel", "transport", "cab", "uber", "ola", "commute", "petrol", "fuel", "bus", "train", "auto", "metro"],
    "bills":              ["bill", "bills", "utilities", "electricity", "recharge", "rent", "subscription", "wifi", "internet", "netflix", "spotify"],
    "shopping":           ["shopping", "amazon", "flipkart", "clothes", "purchase", "buy", "online order", "myntra", "shop"],
    "fraud_alerts":       ["fraud", "suspicious", "anomaly", "unusual", "flagged", "alert", "scam", "fake"],
    "average":            ["average", "avg", "mean", "per transaction", "typical", "normally spend"],
    "largest":            ["largest", "biggest", "highest transaction", "most expensive", "max transaction", "single largest", "maximum expense"],
    "smallest":           ["smallest", "cheapest", "lowest transaction", "minimum", "min transaction", "least expensive"],
    "recent":             ["recent", "last few", "show expenses", "list expenses", "latest", "new expenses", "show me expenses"],
    "save_money":         ["save", "saving", "reduce", "cut costs", "tips", "advice", "how to save", "spend less", "budget tips"],
    "budget_status":      ["budget", "limit", "how much left", "remaining budget", "budget status", "budget check"],
    "trend":              ["trend", "increasing", "decreasing", "pattern", "going up", "going down", "over time", "history", "progress"],
    "best_month":         ["best month", "lowest month", "cheapest month", "good month", "least expensive month"],
    "worst_month":        ["worst month", "highest month", "expensive month", "bad month", "most spent month", "most expensive month"],
    "today":              ["today", "today's expenses", "spent today", "this morning", "this evening"],
    "this_week":          ["this week", "week so far", "weekly", "past 7 days", "last 7 days", "last week"],
    "compare":            ["compare", " vs ", "versus", "difference between", "food vs", "travel vs", "bills vs", "shopping vs"],
    "count":              ["how many", "number of", "count", "transactions count", "how many transactions"],
    "frequency":          ["frequent", "often", "common", "most times", "how often", "recurring", "repeat"],
    "prediction":         ["predict", "forecast", "next month", "will i spend", "expected", "estimate next"],
    "percentage":         ["percent", "percentage", "%", "proportion", "share", "ratio", "what share"],
    "daily_avg":          ["daily", "per day", "each day", "day average", "daily average"],
    "weekend":            ["weekend", "saturday", "sunday", "weekend spending"],
    "weekday":            ["weekday", "monday", "tuesday", "wednesday", "thursday", "friday", "which day"],
}


def detect_intent(msg: str) -> list:
    """Return all matching intent labels for the user message."""
    msg_lower = msg.lower()
    detected = []
    for intent, keywords in INTENT_MAP.items():
        if any(kw in msg_lower for kw in keywords):
            detected.append(intent)
    return detected


# ─────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns needed for analytics."""
    df = df.copy()
    df["amount"]  = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["date"]    = pd.to_datetime(df["date"], errors="coerce")
    df["month"]   = df["date"].dt.to_period("M").astype(str)
    df["day"]     = df["date"].dt.date
    df["weekday"] = df["date"].dt.day_name()
    return df


def _cat_summary(df: pd.DataFrame) -> pd.Series:
    return df.groupby("category")["amount"].sum().sort_values(ascending=False)


def _month_summary(df: pd.DataFrame) -> pd.Series:
    return df.groupby("month")["amount"].sum().sort_values()


def _pct(part, total) -> str:
    if total == 0:
        return "0.0%"
    return f"{(part / total * 100):.1f}%"


def _current_month() -> str:
    return datetime.now().strftime("%Y-%m")


# ─────────────────────────────────────────────────────────────
# RESPONSE BUILDERS — one function per intent, all use real data
# ─────────────────────────────────────────────────────────────

def _resp_greeting(df):
    now = datetime.now()
    hour = now.hour
    if hour < 12:
        time_greet = "Good morning"
    elif hour < 17:
        time_greet = "Good afternoon"
    else:
        time_greet = "Good evening"

    total = df["amount"].sum()
    mdf = df[df["month"] == _current_month()]
    this_month_total = mdf["amount"].sum()
    fraud_count = len(df[df["is_fraud"] == 1])
    top_cat = _cat_summary(df).index[0] if not df.empty else "N/A"

    fraud_note = f"\n- ⚠️ **{fraud_count} suspicious** transaction(s) flagged!" if fraud_count > 0 else ""

    return (
        f"👋 **{time_greet}!** I'm your AI Finance Assistant.\n\n"
        f"📊 **Your quick snapshot:**\n"
        f"- 💰 All-time total: **₹{total:,.2f}**\n"
        f"- 📅 {now.strftime('%B %Y')} so far: **₹{this_month_total:,.2f}**\n"
        f"- 🧾 Total transactions logged: **{len(df)}**\n"
        f"- 🏆 Top spending category: **{top_cat}**{fraud_note}\n\n"
        f"Ask me anything about your finances! Type **help** to see all questions I can answer."
    )


def _resp_help():
    return (
        "🤖 **Complete Question Guide**\n\n"
        "**📊 Spending Summaries**\n"
        "› *What's my total spending?*\n"
        "› *How much did I spend this month?*\n"
        "› *How much did I spend last month?*\n"
        "› *Show category breakdown*\n\n"
        "**🏆 Category Analysis**\n"
        "› *Where did I spend the most?*\n"
        "› *Where did I spend the least?*\n"
        "› *How much on food / travel / bills / shopping?*\n"
        "› *What percentage goes to shopping?*\n"
        "› *Compare food vs travel*\n\n"
        "**📋 Transaction Queries**\n"
        "› *Show my recent expenses*\n"
        "› *What's my largest transaction?*\n"
        "› *What's my smallest transaction?*\n"
        "› *What's my average transaction?*\n"
        "› *How many transactions do I have?*\n"
        "› *What are my most frequent expenses?*\n\n"
        "**📅 Time-Based Analysis**\n"
        "› *What did I spend today?*\n"
        "› *What did I spend this week?*\n"
        "› *What's my daily average?*\n"
        "› *Which day do I spend the most?*\n"
        "› *How much on weekends?*\n"
        "› *What's my spending trend?*\n"
        "› *What was my best / worst month?*\n\n"
        "**⚠️ Alerts & Planning**\n"
        "› *Are there any fraud alerts?*\n"
        "› *What's my budget status?*\n"
        "› *Predict next month's spending*\n\n"
        "**💡 Advice**\n"
        "› *How can I save money?*"
    )


def _resp_total_spending(df):
    total   = df["amount"].sum()
    count   = len(df)
    avg     = df["amount"].mean()
    cats    = _cat_summary(df)
    top_cat = cats.index[0]
    top_amt = cats.iloc[0]
    span    = (df["date"].max() - df["date"].min()).days + 1

    return (
        f"💰 **All-Time Spending Summary**\n\n"
        f"- 🏦 Total: **₹{total:,.2f}**\n"
        f"- 🧾 Transactions: **{count}**\n"
        f"- 📐 Avg per transaction: **₹{avg:,.2f}**\n"
        f"- 📅 Tracking span: **{span} days** ({df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')})\n"
        f"- 📊 Daily average: **₹{total/max(span,1):,.2f}/day**\n"
        f"- 🏆 Top category: **{top_cat}** — ₹{top_amt:,.2f} ({_pct(top_amt, total)} of total)\n\n"
        f"**Category Totals:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f} ({_pct(a, total)})" for c, a in cats.items())
    )


def _resp_this_month(df):
    now = datetime.now()
    mdf = df[df["month"] == _current_month()]

    if mdf.empty:
        return f"📅 No expenses recorded for **{now.strftime('%B %Y')}** yet. Start adding some!"

    total    = mdf["amount"].sum()
    count    = len(mdf)
    avg      = mdf["amount"].mean()
    day_num  = now.day
    daily    = total / day_num
    cats     = _cat_summary(mdf)

    # Compare to last month
    lm = (now.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
    ldf = df[df["month"] == lm]
    compare_line = ""
    if not ldf.empty:
        lm_total = ldf["amount"].sum()
        diff = total - lm_total
        sign = "📈 up" if diff > 0 else "📉 down"
        compare_line = f"\n- vs {ldf['month'].iloc[0]}: {sign} by **₹{abs(diff):,.2f}**"

    # Projection for rest of month
    days_in_month = (now.replace(month=now.month % 12 + 1, day=1) - timedelta(days=1)).day if now.month < 12 else 31
    projected = daily * days_in_month

    return (
        f"📅 **{now.strftime('%B %Y')} Spending** (Day {day_num}/{days_in_month})\n\n"
        f"- 💰 Spent so far: **₹{total:,.2f}**\n"
        f"- 🧾 Transactions: **{count}**\n"
        f"- 📐 Avg transaction: **₹{avg:,.2f}**\n"
        f"- 📊 Daily avg: **₹{daily:,.2f}/day**\n"
        f"- 🔮 Projected month-end: **₹{projected:,.2f}**{compare_line}\n\n"
        f"**This month by category:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f} ({_pct(a, total)})" for c, a in cats.items())
    )


def _resp_last_month(df):
    now = datetime.now()
    lm_dt = now.replace(day=1) - timedelta(days=1)
    lm = lm_dt.strftime("%Y-%m")
    ldf = df[df["month"] == lm]

    if ldf.empty:
        return f"📅 No expenses found for **{lm_dt.strftime('%B %Y')}**."

    total = ldf["amount"].sum()
    count = len(ldf)
    cats  = _cat_summary(ldf)

    return (
        f"📅 **{lm_dt.strftime('%B %Y')} Summary**\n\n"
        f"- 💰 Total: **₹{total:,.2f}**\n"
        f"- 🧾 Transactions: **{count}**\n"
        f"- 📐 Average: **₹{ldf['amount'].mean():,.2f}**\n"
        f"- 🔺 Largest: **₹{ldf['amount'].max():,.2f}** ({ldf.loc[ldf['amount'].idxmax(), 'description']})\n\n"
        f"**By Category:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f} ({_pct(a, total)})" for c, a in cats.items())
    )


def _resp_top_category(df):
    total = df["amount"].sum()
    cats  = _cat_summary(df)
    top_c = cats.index[0]
    top_a = cats.iloc[0]

    # This month's top
    mdf = df[df["month"] == _current_month()]
    m_line = ""
    if not mdf.empty:
        mc = _cat_summary(mdf)
        m_line = f"\n- 📅 This month's top: **{mc.index[0]}** (₹{mc.iloc[0]:,.2f})"

    ranks = "\n".join(
        f"  {i+1}. **{c}**: ₹{a:,.2f} ({_pct(a, total)})"
        for i, (c, a) in enumerate(cats.items())
    )

    return (
        f"🏆 **Where You Spend The Most**\n\n"
        f"- 👑 All-time leader: **{top_c}** — ₹{top_a:,.2f} ({_pct(top_a, total)}){m_line}\n\n"
        f"**Full ranking:**\n{ranks}\n\n"
        f"💡 Reducing **{top_c}** by 20% would save ₹{top_a * 0.20:,.0f} — the biggest single impact you can make!"
    )


def _resp_lowest_category(df):
    total = df["amount"].sum()
    cats  = _cat_summary(df).sort_values()
    low_c = cats.index[0]
    low_a = cats.iloc[0]

    return (
        f"📉 **Where You Spend The Least**\n\n"
        f"- ✅ Lowest: **{low_c}** — ₹{low_a:,.2f} ({_pct(low_a, total)})\n\n"
        f"**Ranked lowest → highest:**\n"
        + "\n".join(f"  {i+1}. {c}: ₹{a:,.2f}" for i, (c, a) in enumerate(cats.items()))
    )


def _resp_category_breakdown(df):
    total = df["amount"].sum()
    cats  = _cat_summary(df)
    mdf   = df[df["month"] == _current_month()]

    lines = []
    for cat, amt in cats.items():
        cdf    = df[df["category"] == cat]
        m_amt  = mdf[mdf["category"] == cat]["amount"].sum() if not mdf.empty else 0
        n      = len(cdf)
        c_avg  = cdf["amount"].mean()
        lines.append(
            f"**{cat}** ({_pct(amt, total)})\n"
            f"  › All-time: ₹{amt:,.2f} | This month: ₹{m_amt:,.2f}\n"
            f"  › {n} transactions | Avg ₹{c_avg:,.2f} each"
        )

    return (
        f"📋 **Complete Category Breakdown**\n\n"
        + "\n\n".join(lines) +
        f"\n\n💰 **Grand Total: ₹{total:,.2f}** across {len(df)} transactions"
    )


def _resp_food(df):
    cdf = df[df["category"] == "Food"]
    if cdf.empty:
        return "🍔 No food expenses found yet. Try adding some!"

    total     = df["amount"].sum()
    cat_total = cdf["amount"].sum()
    count     = len(cdf)
    avg       = cdf["amount"].mean()
    largest   = cdf.loc[cdf["amount"].idxmax()]
    mdf       = cdf[cdf["month"] == _current_month()]
    m_total   = mdf["amount"].sum()
    top_desc  = cdf["description"].value_counts().index[0]

    return (
        f"🍔 **Your Food Spending**\n\n"
        f"- 💰 All-time total: **₹{cat_total:,.2f}** ({_pct(cat_total, total)} of all spending)\n"
        f"- 📅 This month: **₹{m_total:,.2f}**\n"
        f"- 🧾 Transactions: **{count}**\n"
        f"- 📐 Average meal/grocery spend: **₹{avg:,.2f}**\n"
        f"- 🔺 Largest bill: **₹{largest['amount']:,.2f}** on {largest['date'].strftime('%d %b')} ({largest['description']})\n"
        f"- 📝 Most common: *{top_desc}*\n\n"
        f"💡 **Savings tip:** Meal prepping on Sundays and limiting food delivery to twice a week "
        f"can cut your food bill by **₹{cat_total * 0.30 / max(1, len(df['month'].unique())):,.0f}/month**!"
    )


def _resp_travel(df):
    cdf = df[df["category"] == "Travel"]
    if cdf.empty:
        return "🚗 No travel/transport expenses found yet."

    total     = df["amount"].sum()
    cat_total = cdf["amount"].sum()
    count     = len(cdf)
    avg       = cdf["amount"].mean()
    largest   = cdf.loc[cdf["amount"].idxmax()]
    mdf       = cdf[cdf["month"] == _current_month()]
    m_total   = mdf["amount"].sum()
    top_desc  = cdf["description"].value_counts().index[0]

    return (
        f"🚗 **Your Travel / Transport Spending**\n\n"
        f"- 💰 All-time total: **₹{cat_total:,.2f}** ({_pct(cat_total, total)} of all spending)\n"
        f"- 📅 This month: **₹{m_total:,.2f}**\n"
        f"- 🧾 Trips/transactions: **{count}**\n"
        f"- 📐 Avg per trip: **₹{avg:,.2f}**\n"
        f"- 🔺 Most expensive: **₹{largest['amount']:,.2f}** ({largest['description']})\n"
        f"- 📝 Most frequent: *{top_desc}*\n\n"
        f"💡 **Savings tip:** Switching to public transport for even 3 days/week could save "
        f"**₹{avg * 0.4 * 12:,.0f}/year** in cab fares!"
    )


def _resp_bills(df):
    cdf = df[df["category"] == "Bills"]
    if cdf.empty:
        return "💡 No bills/utilities expenses found yet."

    total     = df["amount"].sum()
    cat_total = cdf["amount"].sum()
    count     = len(cdf)
    avg       = cdf["amount"].mean()
    mdf       = cdf[cdf["month"] == _current_month()]
    m_total   = mdf["amount"].sum()

    # List unique bill types
    top_bills = cdf["description"].value_counts().head(4)
    bill_lines = "\n".join(f"  - {d}: {n} times" for d, n in top_bills.items())

    return (
        f"💡 **Your Bills & Utilities**\n\n"
        f"- 💰 All-time total: **₹{cat_total:,.2f}** ({_pct(cat_total, total)} of all spending)\n"
        f"- 📅 This month: **₹{m_total:,.2f}**\n"
        f"- 🧾 Payments: **{count}**\n"
        f"- 📐 Avg payment: **₹{avg:,.2f}**\n\n"
        f"**Most frequent bills:**\n{bill_lines}\n\n"
        f"💡 **Savings tip:** Audit all subscriptions monthly. If you have 4+ subscriptions, "
        f"you could save **₹{min(cat_total * 0.15, 500):,.0f}+/month** by cancelling unused ones!"
    )


def _resp_shopping(df):
    cdf = df[df["category"] == "Shopping"]
    if cdf.empty:
        return "🛍️ No shopping expenses found yet."

    total     = df["amount"].sum()
    cat_total = cdf["amount"].sum()
    count     = len(cdf)
    avg       = cdf["amount"].mean()
    largest   = cdf.loc[cdf["amount"].idxmax()]
    mdf       = cdf[cdf["month"] == _current_month()]
    m_total   = mdf["amount"].sum()
    top_desc  = cdf["description"].value_counts().index[0]

    return (
        f"🛍️ **Your Shopping Expenses**\n\n"
        f"- 💰 All-time total: **₹{cat_total:,.2f}** ({_pct(cat_total, total)} of all spending)\n"
        f"- 📅 This month: **₹{m_total:,.2f}**\n"
        f"- 🧾 Purchases: **{count}**\n"
        f"- 📐 Avg order: **₹{avg:,.2f}**\n"
        f"- 🔺 Biggest purchase: **₹{largest['amount']:,.2f}** ({largest['description']})\n"
        f"- 📝 Most frequent: *{top_desc}*\n\n"
        f"💡 **Savings tip:** Adding items to a wishlist and waiting 48 hours before buying "
        f"eliminates ~30% of impulse purchases — that's potentially ₹{cat_total * 0.30 / max(1, len(df['month'].unique())):,.0f}/month saved!"
    )


def _resp_fraud(df):
    fraud_df = df[df["is_fraud"] == 1]
    count    = len(fraud_df)

    if count == 0:
        avg_n = df["amount"].mean()
        return (
            f"✅ **No Suspicious Transactions Detected!**\n\n"
            f"All **{len(df)}** of your transactions appear normal.\n\n"
            f"- 📐 Your normal avg transaction: ₹{avg_n:,.2f}\n"
            f"- The Isolation Forest algorithm monitors for amounts significantly above your baseline\n\n"
            f"💡 If you add a large, unusual expense, it will be automatically flagged."
        )

    total_n   = df[df["is_fraud"] == 0]["amount"].sum()
    fraud_amt = fraud_df["amount"].sum()
    avg_norm  = df[df["is_fraud"] == 0]["amount"].mean()

    lines = []
    for _, row in fraud_df.iterrows():
        ratio = row["amount"] / avg_norm if avg_norm > 0 else 0
        lines.append(
            f"- ⚠️ **₹{row['amount']:,.2f}** | {row['description']} | "
            f"{row['date'].strftime('%d %b %Y')} | {row['category']} "
            f"({ratio:.1f}× your avg)"
        )

    return (
        f"🚨 **{count} Suspicious Transaction(s) Found!**\n\n"
        + "\n".join(lines) +
        f"\n\n**📊 Context:**\n"
        f"- Your normal avg transaction: ₹{avg_norm:,.2f}\n"
        f"- Flagged transactions total: ₹{fraud_amt:,.2f}\n"
        f"- These are significantly above your usual spending pattern\n\n"
        f"👉 Go to **Fraud Detection** page to re-scan or review all alerts."
    )


def _resp_average(df):
    avg_all  = df["amount"].mean()
    avg_cats = df.groupby("category")["amount"].mean().sort_values(ascending=False)
    mdf      = df[df["month"] == _current_month()]
    avg_m    = mdf["amount"].mean() if not mdf.empty else 0
    median   = df["amount"].median()

    return (
        f"📊 **Your Spending Averages**\n\n"
        f"- 🏦 Overall avg transaction: **₹{avg_all:,.2f}**\n"
        f"- 📊 Median transaction: **₹{median:,.2f}**\n"
        f"- 📅 This month's avg: **₹{avg_m:,.2f}**\n\n"
        f"**Average by category:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f}" for c, a in avg_cats.items()) +
        f"\n\n💡 Any transaction above **₹{avg_all * 2.5:,.0f}** (2.5× your avg) is worth double-checking!"
    )


def _resp_largest(df):
    row = df.loc[df["amount"].idxmax()]
    avg = df["amount"].mean()
    pct_above = ((row["amount"] - avg) / avg) * 100

    return (
        f"🔺 **Your Largest Single Transaction**\n\n"
        f"- 💰 Amount: **₹{row['amount']:,.2f}**\n"
        f"- 📝 Description: {row['description']}\n"
        f"- 📅 Date: {row['date'].strftime('%d %B %Y')}\n"
        f"- 🏷️ Category: {row['category']}\n"
        f"- ⚠️ Fraud flag: {'Yes ⚠️' if row['is_fraud'] == 1 else 'No ✅'}\n\n"
        f"This is **{row['amount']/avg:.1f}× your average** transaction — "
        f"**{pct_above:.0f}% above** your typical spending."
    )


def _resp_smallest(df):
    row = df.loc[df["amount"].idxmin()]
    return (
        f"🔻 **Your Smallest Single Transaction**\n\n"
        f"- 💰 Amount: **₹{row['amount']:,.2f}**\n"
        f"- 📝 Description: {row['description']}\n"
        f"- 📅 Date: {row['date'].strftime('%d %B %Y')}\n"
        f"- 🏷️ Category: {row['category']}"
    )


def _resp_recent(df):
    recent = df.sort_values("date", ascending=False).head(8)
    lines  = []
    for _, row in recent.iterrows():
        flag = " ⚠️" if row["is_fraud"] == 1 else ""
        lines.append(
            f"- **{row['date'].strftime('%d %b')}** | {row['description'][:38]} | "
            f"₹{row['amount']:,.2f} | {row['category']}{flag}"
        )
    return (
        f"📋 **8 Most Recent Expenses**\n\n"
        + "\n".join(lines) +
        f"\n\n_Subtotal: ₹{recent['amount'].sum():,.2f}_"
    )


def _resp_save_money(df):
    total = df["amount"].sum()
    cats  = _cat_summary(df)
    top_c = cats.index[0]
    top_a = cats.iloc[0]

    # Calculate trend
    monthly = _month_summary(df)
    trend_note = ""
    if len(monthly) >= 2:
        delta = monthly.iloc[-1] - monthly.iloc[-2]
        if delta > 0:
            trend_note = f"\n⚠️ Your spending is up ₹{delta:,.2f} vs last month — act now!"
        else:
            trend_note = f"\n✅ Your spending dropped ₹{abs(delta):,.2f} vs last month — keep it up!"

    saving_potential = top_a * 0.20
    months_data = len(monthly)

    category_tips = {
        "Food":     "🍱 Cook at home 5×/week, batch grocery shop, avoid food delivery on weekdays.",
        "Travel":   "🚌 Use monthly transit pass, carpool, combine errands into single trips.",
        "Bills":    "💡 Cancel 1 unused subscription this week, switch to annual plans for 15–20% off.",
        "Shopping": "🛒 Use wishlists + 48-hour rule, buy during sales, compare 3 platforms before purchasing.",
    }

    active_tips = "\n".join(
        f"  {tip}" for cat, tip in category_tips.items() if cat in cats.index
    )

    return (
        f"💡 **Your Personalized Savings Plan**\n"
        f"_(Based on your actual {months_data}-month spending data)_{trend_note}\n\n"
        f"**Your spending distribution:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f} ({_pct(a, total)})" for c, a in cats.items()) +
        f"\n\n**🎯 Biggest lever: {top_c}**\n"
        f"  Cutting {top_c} by 20% saves **₹{saving_potential:,.0f}** — your highest impact action.\n\n"
        f"**Category-specific actions:**\n{active_tips}\n\n"
        f"**📐 The 50/30/20 Rule:**\n"
        f"  - 50% → Needs (bills, groceries, transport)\n"
        f"  - 30% → Wants (dining out, shopping)\n"
        f"  - 20% → Savings & investments\n\n"
        f"**🏆 Your #1 action this week:** Reduce {top_c} spending by ₹{saving_potential/4:,.0f}"
    )


def _resp_budget_status(df, budget_fn=None):
    now   = datetime.now()
    mdf   = df[df["month"] == _current_month()]
    spent = mdf["amount"].sum()
    budget = budget_fn(_current_month()) if budget_fn else 0.0

    if budget <= 0:
        cats = _cat_summary(mdf) if not mdf.empty else pd.Series(dtype=float)
        cat_lines = "\n".join(f"  - {c}: ₹{a:,.2f}" for c, a in cats.items()) if not cats.empty else "  No data"
        return (
            f"📅 **Budget Status: {now.strftime('%B %Y')}**\n\n"
            f"- 💸 Spent so far: **₹{spent:,.2f}**\n"
            f"- ⚠️ No budget set for this month\n\n"
            f"**Breakdown so far:**\n{cat_lines}\n\n"
            f"👉 Go to **Budget Planner** to set a monthly budget and get smart alerts!"
        )

    remaining = budget - spent
    pct       = (spent / budget) * 100
    days_left = (now.replace(month=now.month % 12 + 1, day=1) - timedelta(days=1)).day - now.day if now.month < 12 else 31 - now.day
    safe_daily = remaining / max(days_left, 1)

    if pct >= 100:
        status = f"🚨 **EXCEEDED** — ₹{abs(remaining):,.2f} over budget!"
    elif pct >= 80:
        status = f"⚠️ **WARNING** — {pct:.1f}% used! Only ₹{remaining:,.2f} left."
    elif pct >= 50:
        status = f"🟡 **On Track** — {pct:.1f}% used, ₹{remaining:,.2f} remaining."
    else:
        status = f"✅ **Doing Well** — {pct:.1f}% used, ₹{remaining:,.2f} remaining."

    return (
        f"📅 **Budget Status: {now.strftime('%B %Y')}**\n\n"
        f"- 🎯 Budget: **₹{budget:,.2f}**\n"
        f"- 💸 Spent: **₹{spent:,.2f}**\n"
        f"- 🏦 Remaining: **₹{remaining:,.2f}**\n"
        f"- 📊 Used: **{pct:.1f}%**\n"
        f"- 📅 Days left in month: **{max(days_left, 0)}**\n"
        f"- 💡 Safe to spend: **₹{max(safe_daily, 0):,.2f}/day** to stay within budget\n\n"
        f"**Status:** {status}"
    )


def _resp_trend(df):
    monthly = _month_summary(df)
    if len(monthly) < 2:
        return "📈 Need at least 2 months of data to show a trend. Keep logging expenses!"

    months  = list(monthly.index)
    amounts = list(monthly.values)
    avg     = np.mean(amounts)

    # Linear trend direction
    if len(amounts) >= 3:
        recent_avg  = np.mean(amounts[-2:])
        earlier_avg = np.mean(amounts[:-2])
        change_pct  = ((recent_avg - earlier_avg) / max(earlier_avg, 1)) * 100
    else:
        change_pct = ((amounts[-1] - amounts[0]) / max(amounts[0], 1)) * 100

    if change_pct > 10:
        verdict = f"📈 **Trending UP** (+{change_pct:.1f}%) — spending is rising. Review your habits!"
    elif change_pct < -10:
        verdict = f"📉 **Trending DOWN** ({change_pct:.1f}%) — great improvement!"
    else:
        verdict = f"📊 **Stable** ({change_pct:+.1f}%) — spending is consistent."

    month_lines = "\n".join(
        f"  {'→' if a == max(amounts) else '·'} {m}: ₹{a:,.2f}"
        + (" ← highest" if a == max(amounts) else " ← lowest" if a == min(amounts) else "")
        for m, a in zip(months, amounts)
    )

    return (
        f"📈 **Your Spending Trend**\n\n"
        f"**Monthly history:**\n{month_lines}\n\n"
        f"**{verdict}**\n\n"
        f"- 📊 Monthly avg: **₹{avg:,.2f}**\n"
        f"- 🔺 Peak month: **{monthly.idxmax()}** (₹{monthly.max():,.2f})\n"
        f"- 🔻 Best month: **{monthly.idxmin()}** (₹{monthly.min():,.2f})\n"
        f"- 📐 Variance: ₹{np.std(amounts):,.2f}"
    )


def _resp_best_month(df):
    monthly = _month_summary(df)
    if monthly.empty:
        return "📅 Not enough data yet."
    best   = monthly.idxmin()
    best_a = monthly.min()
    avg    = monthly.mean()
    saved  = avg - best_a

    return (
        f"🌟 **Your Best Month (Lowest Spending)**\n\n"
        f"- 📅 Month: **{best}**\n"
        f"- 💰 Total: **₹{best_a:,.2f}**\n"
        f"- 📊 Monthly avg: ₹{avg:,.2f}\n"
        f"- 🏆 Saved **₹{saved:,.2f}** vs average that month!\n\n"
        f"💡 Try to replicate {best}'s habits — what did you do differently?"
    )


def _resp_worst_month(df):
    monthly = _month_summary(df)
    if monthly.empty:
        return "📅 Not enough data yet."
    worst   = monthly.idxmax()
    worst_a = monthly.max()
    avg     = monthly.mean()
    excess  = worst_a - avg

    return (
        f"😬 **Your Most Expensive Month**\n\n"
        f"- 📅 Month: **{worst}**\n"
        f"- 💰 Total: **₹{worst_a:,.2f}**\n"
        f"- 📊 Monthly avg: ₹{avg:,.2f}\n"
        f"- ⚠️ Spent **₹{excess:,.2f} above average** that month\n\n"
        f"💡 Review {worst} — was it a one-off event or a habit to fix?"
    )


def _resp_today(df):
    today = datetime.now().date()
    tdf   = df[df["day"] == today]

    if tdf.empty:
        return f"📅 No expenses logged **today** ({today.strftime('%d %B %Y')}) yet."

    total = tdf["amount"].sum()
    lines = [
        f"- {row['description'][:40]} | ₹{row['amount']:,.2f} | {row['category']}"
        for _, row in tdf.iterrows()
    ]
    return (
        f"📅 **Today ({today.strftime('%d %B %Y')})**\n\n"
        + "\n".join(lines) +
        f"\n\n💰 **Today's total: ₹{total:,.2f}**"
    )


def _resp_this_week(df):
    today     = datetime.now().date()
    week_ago  = today - timedelta(days=7)
    wdf       = df[df["day"] >= week_ago]

    if wdf.empty:
        return "📅 No expenses in the **last 7 days**."

    total = wdf["amount"].sum()
    cats  = _cat_summary(wdf)

    return (
        f"📅 **Last 7 Days** ({week_ago.strftime('%d %b')} → {today.strftime('%d %b')})\n\n"
        f"- 💰 Total: **₹{total:,.2f}**\n"
        f"- 🧾 Transactions: **{len(wdf)}**\n"
        f"- 📐 Daily average: **₹{total/7:,.2f}**\n\n"
        f"**By Category:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f}" for c, a in cats.items())
    )


def _resp_compare(df, msg):
    cat_map = {
        "food": "Food", "eating": "Food", "groceries": "Food",
        "travel": "Travel", "transport": "Travel", "cab": "Travel",
        "bills": "Bills", "utilities": "Bills",
        "shopping": "Shopping", "shop": "Shopping",
    }
    found = []
    for k, v in cat_map.items():
        if k in msg.lower() and v not in found:
            found.append(v)

    if len(found) < 2:
        cats  = _cat_summary(df)
        total = df["amount"].sum()
        lines = [f"  **{c}**: ₹{a:,.2f} ({_pct(a, total)})" for c, a in cats.items()]
        return "📊 **All Categories Compared:**\n\n" + "\n".join(lines) + f"\n\n💰 Total: ₹{total:,.2f}"

    c1, c2 = found[0], found[1]
    a1 = df[df["category"] == c1]["amount"].sum()
    a2 = df[df["category"] == c2]["amount"].sum()
    n1 = len(df[df["category"] == c1])
    n2 = len(df[df["category"] == c2])
    winner = c1 if a1 > a2 else c2
    diff   = abs(a1 - a2)
    ratio  = max(a1, a2) / max(min(a1, a2), 1)

    return (
        f"⚖️ **{c1} vs {c2}**\n\n"
        f"- 🔵 **{c1}**: ₹{a1:,.2f} ({n1} transactions, avg ₹{a1/max(n1,1):,.2f})\n"
        f"- 🟠 **{c2}**: ₹{a2:,.2f} ({n2} transactions, avg ₹{a2/max(n2,1):,.2f})\n\n"
        f"🏆 You spend **{ratio:.1f}× more** on **{winner}** — a difference of ₹{diff:,.2f}\n\n"
        f"💡 Cutting **{winner}** spending by 15% would save ₹{max(a1,a2)*0.15:,.0f}!"
    )


def _resp_count(df):
    total    = len(df)
    by_cat   = df.groupby("category").size().sort_values(ascending=False)
    mdf      = df[df["month"] == _current_month()]
    m_count  = len(mdf)
    monthly  = df.groupby("month").size()
    avg_pm   = monthly.mean()

    return (
        f"🔢 **Transaction Count**\n\n"
        f"- 🧾 All-time: **{total}** transactions\n"
        f"- 📅 This month: **{m_count}**\n"
        f"- 📐 Monthly average: **{avg_pm:.1f}** transactions\n\n"
        f"**By Category:**\n"
        + "\n".join(f"  - {c}: {n} ({_pct(n, total)})" for c, n in by_cat.items())
    )


def _resp_frequency(df):
    freq = df.groupby("description").size().sort_values(ascending=False).head(6)
    lines = [
        f"  {i+1}. *{desc}* — **{n} times**"
        for i, (desc, n) in enumerate(freq.items())
    ]
    return (
        f"🔄 **Your Most Frequent Expenses**\n\n"
        + "\n".join(lines) +
        f"\n\n💡 Recurring expenses are great candidates for negotiation or bulk-deal discounts!"
    )


def _resp_prediction(df):
    from models.predictor import predict_next_month
    monthly = _month_summary(df)
    now     = datetime.now()

    if len(monthly) < 2:
        return "🔮 Need at least **2 months** of data to predict. Keep logging expenses!"

    predicted   = predict_next_month(list(monthly.values))
    last_actual = monthly.iloc[-1]
    diff        = predicted - last_actual
    avg_monthly = monthly.mean()
    next_m_name = (now.replace(day=1) + timedelta(days=32)).strftime("%B %Y")

    return (
        f"🔮 **Spending Forecast: {next_m_name}**\n\n"
        f"- 📊 Model trained on: **{len(monthly)} months** of your data\n"
        f"- 🔮 Predicted total: **₹{predicted:,.2f}**\n"
        f"- 📅 Last month's actual: ₹{last_actual:,.2f}\n"
        f"- 📐 Historical avg: ₹{avg_monthly:,.2f}\n"
        f"- {'📈' if diff > 0 else '📉'} Predicted change: {'▲' if diff > 0 else '▼'} ₹{abs(diff):,.2f}\n\n"
        f"{'⚠️ Spending is trending upward — tighten your budget now!' if diff > 200 else '✅ Spending looks stable or declining — great trajectory!'}\n\n"
        f"👉 See the full **Prediction** page for a visual chart."
    )


def _resp_percentage(df, msg):
    total = df["amount"].sum()
    cats  = _cat_summary(df)
    cat_map = {"food": "Food", "travel": "Travel", "bills": "Bills", "shopping": "Shopping"}

    for kw, cat in cat_map.items():
        if kw in msg.lower() and cat in cats.index:
            pct_val = (cats[cat] / total) * 100
            warn = ""
            if cat == "Food" and pct_val > 40:
                warn = "\n⚠️ Food is >40% of your budget — consider cooking more at home."
            elif cat == "Shopping" and pct_val > 30:
                warn = "\n⚠️ Shopping is >30% of your budget — review discretionary spending."
            return (
                f"📊 **{cat}: {pct_val:.1f}% of total spending**\n\n"
                f"- ₹{cats[cat]:,.2f} on {cat} out of ₹{total:,.2f} total{warn}\n\n"
                f"**All categories:**\n"
                + "\n".join(f"  - {c}: {_pct(a, total)}" for c, a in cats.items())
            )

    return (
        f"📊 **Spending Percentages**\n\n"
        + "\n".join(f"  - **{c}**: {_pct(a, total)} (₹{a:,.2f})" for c, a in cats.items()) +
        f"\n\n💰 Total: ₹{total:,.2f}"
    )


def _resp_daily_avg(df):
    span      = (df["date"].max() - df["date"].min()).days + 1
    daily_avg = df["amount"].sum() / max(span, 1)
    proj_mth  = daily_avg * 30

    now   = datetime.now()
    mdf   = df[df["month"] == _current_month()]
    m_day = mdf["amount"].sum() / now.day if not mdf.empty else 0

    return (
        f"📊 **Daily Average Spending**\n\n"
        f"- 📐 Overall avg: **₹{daily_avg:,.2f}/day**\n"
        f"- 📅 This month avg: **₹{m_day:,.2f}/day**\n"
        f"- 📈 At current pace: **₹{proj_mth:,.2f}/month**\n\n"
        f"💡 Cutting just **₹50/day** saves:\n"
        f"  - ₹1,500/month\n"
        f"  - ₹18,000/year\n"
        f"  - That's a solid emergency fund!"
    )


def _resp_weekend(df):
    wdf  = df[df["weekday"].isin(["Saturday", "Sunday"])]
    wkdf = df[~df["weekday"].isin(["Saturday", "Sunday"])]

    if wdf.empty:
        return "📅 No weekend expenses found yet."

    w_total = wdf["amount"].sum()
    d_total = wkdf["amount"].sum()
    w_avg   = wdf["amount"].mean()
    d_avg   = wkdf["amount"].mean()
    w_cats  = _cat_summary(wdf)

    return (
        f"📅 **Weekend vs Weekday Spending**\n\n"
        f"**🏖️ Weekends:**\n"
        f"  - Total: ₹{w_total:,.2f} | {len(wdf)} transactions | Avg ₹{w_avg:,.2f}\n\n"
        f"**💼 Weekdays:**\n"
        f"  - Total: ₹{d_total:,.2f} | {len(wkdf)} transactions | Avg ₹{d_avg:,.2f}\n\n"
        f"**Weekend spending by category:**\n"
        + "\n".join(f"  - {c}: ₹{a:,.2f}" for c, a in w_cats.items()) +
        f"\n\n{'⚠️ Weekend spending is higher — social activities and shopping drive the difference.' if w_avg > d_avg else '💼 Weekday expenses are higher — commuting and work-related costs add up.'}"
    )


def _resp_weekday_pattern(df):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_day    = df.groupby("weekday")["amount"].sum()
    by_day    = by_day.reindex([d for d in day_order if d in by_day.index])

    if by_day.empty:
        return "📅 Not enough data for weekday analysis."

    top_day = by_day.idxmax()
    low_day = by_day.idxmin()

    lines = [
        f"  {'▶' if day == top_day else ' '} {day}: ₹{amt:,.2f}"
        + (" ← most" if day == top_day else " ← least" if day == low_day else "")
        for day, amt in by_day.items()
    ]

    return (
        f"📅 **Spending by Day of Week**\n\n"
        + "\n".join(lines) +
        f"\n\n🏆 Most spending: **{top_day}**\n"
        f"✅ Most frugal: **{low_day}**\n\n"
        f"💡 Plan big purchases for **{low_day}** (your lowest spend day) to stay disciplined!"
    )


# ─────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────

def get_chatbot_response(user_input: str, df: pd.DataFrame, budget_fn=None) -> str:
    """
    Main chatbot entry point.

    Parameters:
        user_input - raw text from user
        df         - full expense DataFrame from get_all_expenses()
        budget_fn  - optional: get_budget(month_str) → float

    Returns:
        Markdown-formatted response with real live data from SQLite.
    """
    if df.empty:
        return (
            "🤖 No expenses found in your database yet.\n\n"
            "👉 Go to **Add Expense** to log your first transaction, "
            "then come back and ask me anything!"
        )

    df      = _prep(df)
    msg     = user_input.strip()
    intents = detect_intent(msg)

    # Priority routing — order matters
    if "greeting"           in intents: return _resp_greeting(df)
    if "help"               in intents: return _resp_help()
    if "compare"            in intents: return _resp_compare(df, msg)
    if "fraud_alerts"       in intents: return _resp_fraud(df)
    if "prediction"         in intents: return _resp_prediction(df)
    if "budget_status"      in intents: return _resp_budget_status(df, budget_fn)
    if "save_money"         in intents: return _resp_save_money(df)
    if "trend"              in intents: return _resp_trend(df)
    if "worst_month"        in intents: return _resp_worst_month(df)
    if "best_month"         in intents: return _resp_best_month(df)
    if "today"              in intents: return _resp_today(df)
    if "this_week"          in intents: return _resp_this_week(df)
    if "weekend"            in intents: return _resp_weekend(df)
    if "weekday"            in intents: return _resp_weekday_pattern(df)
    if "largest"            in intents: return _resp_largest(df)
    if "smallest"           in intents: return _resp_smallest(df)
    if "frequency"          in intents: return _resp_frequency(df)
    if "count"              in intents: return _resp_count(df)
    if "daily_avg"          in intents: return _resp_daily_avg(df)
    if "percentage"         in intents: return _resp_percentage(df, msg)
    if "food"               in intents: return _resp_food(df)
    if "travel"             in intents: return _resp_travel(df)
    if "bills"              in intents: return _resp_bills(df)
    if "shopping"           in intents: return _resp_shopping(df)
    if "average"            in intents: return _resp_average(df)
    if "recent"             in intents: return _resp_recent(df)
    if "top_category"       in intents: return _resp_top_category(df)
    if "lowest_category"    in intents: return _resp_lowest_category(df)
    if "category_breakdown" in intents: return _resp_category_breakdown(df)
    if "last_month"         in intents: return _resp_last_month(df)
    if "this_month"         in intents: return _resp_this_month(df)
    if "total_spending"     in intents: return _resp_total_spending(df)

    return (
        "🤔 I didn't quite catch that. Here's what I can help with:\n\n"
        "- *What's my total spending?*\n"
        "- *Show this month's expenses*\n"
        "- *Where do I spend the most?*\n"
        "- *How can I save money?*\n"
        "- *Are there any fraud alerts?*\n"
        "- *Predict next month's spending*\n"
        "- *Compare food vs travel*\n\n"
        "Type **help** for the complete list of questions I can answer!"
    )
