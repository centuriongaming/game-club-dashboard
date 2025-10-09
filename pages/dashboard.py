# pages/dashboard.py
import streamlit as st
import pandas as pd

# --- Authentication Check ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Dashboard")
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

# CORRECTED QUERIES for KPIs
total_ratings_query = "SELECT COUNT(score) AS total FROM ratings;"
avg_score_query = "SELECT AVG(score) AS average FROM ratings WHERE score IS NOT NULL;"
participation_query = """
    SELECT 
        (COUNT(r.score)::FLOAT / (SELECT COUNT(*) FROM critics) / (SELECT COUNT(*) FROM games WHERE upcoming IS FALSE))
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
col3.metric("Group Participation", f"{participation_rate:.1%}")


# --- 2. Recent Activity Feed ---
st.header("Recent Activity")
# CORRECTED QUERY for Recent Activity
recent_activity_query = """
    SELECT
        g.game_name,
        c.critic_name,
        r.score
    FROM ratings r
    JOIN games g ON r.game_id = g.id
    JOIN critics c ON r.critic_id = c.id
    WHERE r.score IS NOT NULL
    ORDER BY r.id DESC
    LIMIT 5;
"""
recent_activity_df = conn.query(recent_activity_query)
st.dataframe(
    recent_activity_df,
    column_aliases={"game_name": "Game", "critic_name": "Critic", "score": "Score"},
    hide_index=True
)


# --- 3. Upcoming Games ---
st.header("Upcoming Games")
st.write("These games have been nominated but are not yet released or available for review.")

# CORRECTED QUERY for Upcoming Games to show nominator's name
upcoming_games_query = """
    SELECT 
        g.game_name, 
        c.critic_name AS nominated_by, 
        g.release_date
    FROM games g
    LEFT JOIN critics c ON g.nominated_by = c.id
    WHERE g.upcoming IS TRUE
    ORDER BY g.release_date ASC;
"""
upcoming_games_df = conn.query(upcoming_games_query)

if upcoming_games_df.empty:
    st.info("There are currently no games marked as upcoming.")
else:
    st.dataframe(
        upcoming_games_df,
        column_aliases={"game_name": "Game", "nominated_by": "Nominated By", "release_date": "Release Date"},
        hide_index=True
    )

# --- Log Out Button ---
if st.button("Log out"):
    del st.session_state["password_correct"]
    st.rerun()