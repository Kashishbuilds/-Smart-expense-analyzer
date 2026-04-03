# utils/charts.py
# Helper functions for creating Plotly charts used in the dashboard

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def category_pie_chart(df: pd.DataFrame):
    """
    Create a pie chart showing spending distribution by category.

    Parameters:
        df - expense DataFrame with 'category' and 'amount' columns

    Returns:
        Plotly Figure
    """
    cat_totals = df.groupby("category")["amount"].sum().reset_index()
    cat_totals.columns = ["Category", "Amount"]

    fig = px.pie(
        cat_totals,
        names="Category",
        values="Amount",
        title="💸 Spending by Category",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.35,  # donut style looks cleaner
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        title_font_size=18,
        showlegend=True,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def monthly_bar_chart(df: pd.DataFrame):
    """
    Create a bar chart showing total spending per month.

    Parameters:
        df - expense DataFrame with 'date' and 'amount' columns

    Returns:
        Plotly Figure
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.groupby("month")["amount"].sum().reset_index()
    monthly.columns = ["Month", "Total Spending (₹)"]
    monthly = monthly.sort_values("Month")

    fig = px.bar(
        monthly,
        x="Month",
        y="Total Spending (₹)",
        title="📅 Monthly Spending Trend",
        color="Total Spending (₹)",
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        margin=dict(t=60, b=60),
    )
    return fig


def category_bar_chart(df: pd.DataFrame):
    """
    Create a horizontal bar chart of category-wise spending.

    Returns:
        Plotly Figure
    """
    cat_totals = df.groupby("category")["amount"].sum().reset_index()
    cat_totals.columns = ["Category", "Amount (₹)"]
    cat_totals = cat_totals.sort_values("Amount (₹)", ascending=True)

    fig = px.bar(
        cat_totals,
        x="Amount (₹)",
        y="Category",
        orientation="h",
        title="📊 Category-wise Spending",
        color="Amount (₹)",
        color_continuous_scale="Viridis",
        text_auto=True,
    )
    fig.update_layout(
        title_font_size=18,
        coloraxis_showscale=False,
        margin=dict(t=60, b=20),
    )
    return fig


def line_chart_monthly(df: pd.DataFrame):
    """
    Create a line chart for monthly spending trend.

    Returns:
        Plotly Figure
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    monthly = df.groupby("month")["amount"].sum().reset_index()
    monthly.columns = ["Month", "Total (₹)"]
    monthly = monthly.sort_values("Month")

    fig = px.line(
        monthly,
        x="Month",
        y="Total (₹)",
        title="📈 Spending Over Time",
        markers=True,
    )
    fig.update_traces(line_color="#6366f1", line_width=3, marker_size=8)
    fig.update_layout(
        title_font_size=18,
        xaxis_tickangle=-45,
        margin=dict(t=60, b=60),
    )
    return fig


def budget_gauge(spent: float, budget: float):
    """
    Create a gauge chart showing budget utilisation.

    Parameters:
        spent  - amount spent so far this month
        budget - monthly budget limit

    Returns:
        Plotly Figure
    """
    pct = (spent / budget * 100) if budget > 0 else 0

    # Choose colour based on how much budget is used
    if pct >= 100:
        bar_color = "#ef4444"   # red
    elif pct >= 80:
        bar_color = "#f97316"   # orange
    else:
        bar_color = "#22c55e"   # green

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=spent,
        delta={"reference": budget, "suffix": " ₹"},
        title={"text": f"Budget Used: {pct:.1f}%", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, max(budget * 1.2, spent * 1.1)], "ticksuffix": "₹"},
            "bar": {"color": bar_color},
            "steps": [
                {"range": [0, budget * 0.8], "color": "#d1fae5"},
                {"range": [budget * 0.8, budget], "color": "#fef3c7"},
                {"range": [budget, max(budget * 1.2, spent * 1.1)], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": budget,
            },
        },
        number={"suffix": " ₹"},
    ))
    fig.update_layout(margin=dict(t=40, b=20, l=30, r=30), height=280)
    return fig
