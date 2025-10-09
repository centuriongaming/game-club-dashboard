# pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Authentication Check ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard Overview")
st.write("A high-level summary of your group's rating activity.")

# --- Database Connection ---
try:
    conn = st.connection("mydb", type="sql")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# --- 1. Key Performance Indicators (KPIs) ---
st.header("Key Metrics")

# SQL queries for KPIs
total_ratings_query = "SELECT COUNT(score) AS total FROM ratings;"
avg_score_query = "SELECT AVG(score) AS average FROM ratings WHERE score IS NOT NULL;"
participation_query = """
    SELECT 
        (COUNT(r.score)::FLOAT / (SELECT COUNT(*) FROM critics) / (SELECT COUNT(*) FROM games WHERE upcoming IS FALSE)) * 100
    AS rate FROM ratings r;
"""

# Fetch data
total_ratings = conn.query(total_ratings_query)['total'][0]
avg_score = conn.query(avg_score_query)['average'][0]
participation_rate = conn.query(participation_query)['rate'][0]

# Display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Ratings Given", f"{total_ratings}")
col2.metric("Overall Average Score", f"{avg_score:.2f}")
col3.metric("Group Participation", f"{participation_rate:.1f}%")

# --- 2. Visual Breakdowns ---
st.header("Visual Breakdowns")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nominations by Critic")
    # Query for nomination pie chart
    nomination_query = """
        SELECT c.critic_name, COUNT(g.id) AS nomination_count
        FROM games g
        JOIN critics c ON g.nominated_by = c.id
        GROUP BY c.critic_name;
    """
    nomination_df = conn.query(nomination_query)
    
    # Create Pie Chart
    fig = go.Figure(data=[go.Pie(labels=nomination_df['critic_name'], values=nomination_df['nomination_count'], hole=.3)])
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Score Distribution by Critic")
    # Query for binned ratings
    binned_ratings_query = """
        SELECT 
            c.critic_name,
            CASE
                WHEN r.score >= 9 THEN '9-10 (Excellent)'
                WHEN r.score >= 7 THEN '7-8.9 (Good)'
                WHEN r.score >= 5 THEN '5-6.9 (Average)'
                ELSE '0-4.9 (Poor)'
            END as score_bin,
            COUNT(r.id) as rating_count
        FROM ratings r
        JOIN critics c ON r.critic_id = c.id
        WHERE r.score IS NOT NULL
        GROUP BY c.critic_name, score_bin;
    """
    binned_df = conn.query(binned_ratings_query)
    
    # Pivot for stacked bar chart and display
    if not binned_df.empty:
        binned_pivot = binned_df.pivot(index='score_bin', columns='critic_name', values='rating_count').fillna(0)
        st.bar_chart(binned_pivot)
    else:
        st.info("No ratings available for the stacked bar chart.")


# --- 3. Controversy Analysis ---
st.header("Controversy Analysis")
st.write("Controversy is measured by the standard deviation of scores. A high value means critics disagreed widely.")

# Query for controversy (standard deviation)
controversy_query = """
    SELECT 
        g.game_name,
        STDDEV(r.score) as controversy_score,
        COUNT(r.score) as number_of_ratings
    FROM ratings r
    JOIN games g ON r.game_id = g.id
    GROUP BY g.game_name
    HAVING COUNT(r.score) > 1 -- Only include games with more than one rating
    ORDER BY controversy_score DESC;
"""
controversy_df = conn.query(controversy_query)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Most Controversial Games")
    st.dataframe(
        controversy_df.head(5),
        column_config={"controversy_score": st.column_config.NumberColumn(format="%.3f")},
        hide_index=True
    )
with col2:
    st.subheader("Least Controversial Games")
    st.dataframe(
        controversy_df.sort_values("controversy_score", ascending=True).head(5),
        column_config={"controversy_score": st.column_config.NumberColumn(format="%.3f")},
        hide_index=True
    )


# --- 4. Upcoming Games ---
st.header("Upcoming Games")
upcoming_games_query = """
    SELECT 
        g.game_name, 
        c.critic_name AS nominated_by
    FROM games g
    LEFT JOIN critics c ON g.nominated_by = c.id
    WHERE g.upcoming IS TRUE
    ORDER BY g.game_name ASC;
"""
upcoming_games_df = conn.query(upcoming_games_query)

if upcoming_games_df.empty:
    st.info("There are currently no games marked as upcoming.")
else:
    upcoming_games_df = upcoming_games_df.rename(columns={"game_name": "Game", "nominated_by": "Nominated By"})
    st.dataframe(upcoming_games_df, hide_index=True)

# --- Log Out Button ---
if st.button("Log out"):
    del st.session_state["password_correct"]
    st.rerun()