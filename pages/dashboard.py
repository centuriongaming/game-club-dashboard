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

# --- 1. Critic Participation ---
st.header("Critic Participation")
st.write("This table shows how many games each critic has rated out of the total available.")

# CORRECTED QUERY: Changed c.critic_id to c.id in the JOIN clause
critic_participation_query = """
    SELECT
        c.critic_name,
        COUNT(r.score) AS ratings_given,
        (SELECT COUNT(*) FROM games WHERE upcoming IS FALSE) AS total_games,
        AVG(r.score) AS average_score
    FROM critics c
    LEFT JOIN ratings r ON c.id = r.critic_id
    GROUP BY c.critic_name
    ORDER BY ratings_given DESC;
"""
critic_participation_df = conn.query(critic_participation_query)
critic_participation_df['participation_rate'] = critic_participation_df['ratings_given'] / critic_participation_df['total_games']
st.dataframe(
    critic_participation_df,
    column_config={
        "average_score": st.column_config.ProgressColumn(
            "Average Score",
            format="%.2f",
            min_value=0,
            max_value=10,
        ),
        "participation_rate": st.column_config.ProgressColumn(
            "Participation Rate",
            format="%.1f%%",
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True
)


# --- (Rest of the file remains the same) ---

# --- 2. Recent Activity Feed ---
st.header("Recent Activity")
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
recent_activity_df = recent_activity_df.rename(
    columns={"game_name": "Game", "critic_name": "Critic", "score": "Score"}
)
st.dataframe(
    recent_activity_df,
    hide_index=True
)

# --- 3. Upcoming Games ---
st.header("Upcoming Games")
st.write("These games have been nominated but are not yet released or available for review.")

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
    upcoming_games_df = upcoming_games_df.rename(
        columns={"game_name": "Game", "nominated_by": "Nominated By", "release_date": "Release Date"}
    )
    st.dataframe(
        upcoming_games_df,
        hide_index=True
    )

# --- Log Out Button ---
if st.button("Log out"):
    del st.session_state["password_correct"]
    st.rerun()