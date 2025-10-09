# pages/dashboard.py
import streamlit as st
import pandas as pd

# --- Authentication Check ---
# Assumes you have a utils.py file with check_auth() and get_db_connection()
# If not, you can revert to the original individual checks.
try:
    from utils import check_auth, get_db_connection
    check_auth()
    conn = get_db_connection()
except (ImportError, ModuleNotFoundError):
    # Fallback for stand-alone execution or if utils.py is not available
    if not st.session_state.get("password_correct", False):
        st.error("You need to log in first.")
        st.stop()
    try:
        conn = st.connection("mydb", type="sql")
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()


# --- Page Configuration ---
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 Dashboard & Game Rankings")

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

# --- Main Data Query for Rankings ---
main_query = """
WITH global_stats AS (
    SELECT
        2 AS C,
        (SELECT AVG(score) FROM ratings WHERE score IS NOT NULL) as m,
        (PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY score)) as typical_downside_deviation,
        AVG(CASE WHEN g.upcoming IS FALSE THEN 1 ELSE 0 END) as avg_participation_rate
    FROM games g
    LEFT JOIN ratings r ON g.id = r.game_id
),
game_stats AS (
    SELECT
        g.game_name,
        COUNT(r.score) as n,
        AVG(r.score) as x_bar,
        (SELECT COUNT(*) FROM critics) AS total_critics
    FROM games g
    LEFT JOIN ratings r ON g.id = r.game_id
    WHERE g.upcoming IS FALSE AND r.score IS NOT NULL
    GROUP BY g.game_name
)
SELECT
    gs.game_name,
    gs.x_bar as average_score,
    gs.n as number_of_ratings,
    ( (gs.n * gs.x_bar) + (glob.C * glob.m) ) / ( gs.n + glob.C ) AS bayesian_average,
    ( (gs.n * gs.x_bar) + (glob.C * glob.m) ) / ( gs.n + glob.C ) - 
    ( 
        glob.typical_downside_deviation * (1 / (1 + EXP(10 * ( (gs.n::FLOAT / gs.total_critics::FLOAT) - glob.avg_participation_rate ))))
    )
    AS final_adjusted_score
FROM game_stats gs, global_stats glob;
"""
rankings_df = conn.query(main_query)

# --- Best & Worst Games Showcase ---
st.header("Top & Bottom Games")
best_unadjusted = rankings_df.loc[rankings_df['average_score'].idxmax()]
worst_unadjusted = rankings_df.loc[rankings_df['average_score'].idxmin()]
best_adjusted = rankings_df.loc[rankings_df['final_adjusted_score'].idxmax()]
worst_adjusted = rankings_df.loc[rankings_df['final_adjusted_score'].idxmin()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("Best (Unadjusted)")
    st.metric(label=best_unadjusted['game_name'], value=f"{best_unadjusted['average_score']:.2f}")
with col2:
    st.subheader("Worst (Unadjusted)")
    st.metric(label=worst_unadjusted['game_name'], value=f"{worst_unadjusted['average_score']:.2f}")
with col3:
    st.subheader("Best (Adjusted)")
    st.metric(label=best_adjusted['game_name'], value=f"{best_adjusted['final_adjusted_score']:.2f}")
with col4:
    st.subheader("Worst (Adjusted)")
    st.metric(label=worst_adjusted['game_name'], value=f"{worst_adjusted['final_adjusted_score']:.2f}")

st.divider()

# --- Full Adjusted Rankings Table ---
st.header("Full Adjusted Rankings")
st.write("Click on a column header to sort the rankings.")
st.dataframe(
    rankings_df[['game_name', 'average_score', 'bayesian_average', 'final_adjusted_score', 'number_of_ratings']],
    column_config={
        "game_name": "Game", 
        "average_score": "Unadjusted Avg.", 
        "bayesian_average": "Certainty Score", 
        "final_adjusted_score": "Final Score", 
        "number_of_ratings": "Ratings"
    },
    hide_index=True, 
    use_container_width=True
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
    st.dataframe(upcoming_games_df, hide_index=True, use_container_width=True)

# --- Log Out Button ---
if st.button("Log out"):
    st.session_state["password_correct"] = False
    st.rerun()