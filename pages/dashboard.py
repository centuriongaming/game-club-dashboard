# pages/dashboard.py
import streamlit as st
import pandas as pd

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

# --- Key Performance Indicators (KPIs) ---
st.header("Key Metrics")
total_ratings_query = "SELECT COUNT(score) AS total FROM ratings;"
avg_score_query = "SELECT AVG(score) AS average FROM ratings WHERE score IS NOT NULL;"
participation_query = """
    SELECT 
        (COUNT(r.score)::FLOAT / (SELECT COUNT(*) FROM critics) / (SELECT COUNT(*) FROM games WHERE upcoming IS FALSE)) * 100
    AS rate FROM ratings r;
"""
total_ratings = conn.query(total_ratings_query)['total'][0]
avg_score = conn.query(avg_score_query)['average'][0]
participation_rate = conn.query(participation_query)['rate'][0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Ratings Given", f"{total_ratings}")
col2.metric("Overall Average Score", f"{avg_score:.2f}")
col3.metric("Group Participation", f"{participation_rate:.1f}%")

st.divider()

# --- Static Game Rankings ---
st.header("Game Leaderboards")

# --- Adjusted Rankings (Bayesian Average) ---
st.subheader("Game Rankings by Adjusted Score")
with st.expander("How is 'Adjusted Score' calculated?"):
    st.write("""
        This ranking uses a **Bayesian Average**. It balances a game's raw score with the statistical certainty of that score, 
        using a 'credibility constant' of **C=2**, as determined by your group's poll.
    """)

adjusted_rankings_query = """
WITH global_stats AS (
    SELECT (SELECT AVG(score) FROM ratings WHERE score IS NOT NULL) as m
),
game_stats AS (
    SELECT
        g.game_name,
        COUNT(r.score) as n,
        AVG(r.score) as x_bar
    FROM games g
    JOIN ratings r ON g.id = r.game_id
    WHERE g.upcoming IS FALSE AND r.score IS NOT NULL
    GROUP BY g.game_name
)
SELECT
    gs.game_name,
    gs.x_bar as average_score,
    gs.n as number_of_ratings,
    ( (gs.n * gs.x_bar) + (2 * glob.m) ) / ( gs.n + 2 ) AS adjusted_score
FROM game_stats gs, global_stats glob;
"""
adjusted_rankings_df = conn.query(adjusted_rankings_query)

st.write("Click a column header to sort.")
st.dataframe(
    adjusted_rankings_df.sort_values("adjusted_score", ascending=False),
    use_container_width=True,
    hide_index=True
)

# --- Controversy Analysis ---
st.subheader("Game Controversy")
st.write("Controversy is measured by the standard deviation of scores. A high value means critics disagreed widely.")
controversy_query = """
    SELECT 
        g.game_name,
        STDDEV(r.score) as controversy_score,
        COUNT(r.score) as number_of_ratings
    FROM ratings r
    JOIN games g ON r.game_id = g.id
    GROUP BY g.game_name
    HAVING COUNT(r.score) > 1;
"""
controversy_df = conn.query(controversy_query)

st.write("Click a column header to sort.")
st.dataframe(
    controversy_df.sort_values("controversy_score", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.divider()

# --- Upcoming Games ---
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