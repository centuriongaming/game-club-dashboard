# pages/dashboard.py
import streamlit as st
import pandas as pd

# --- Authentication and DB Connection ---
try:
    from utils import check_auth, get_db_connection
    check_auth()
    conn = get_db_connection()
except (ImportError, ModuleNotFoundError):
    # Fallback for stand-alone execution
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
st.title("Dashboard & Game Rankings")

# --- Key Performance Indicators ---
with st.container(border=True):
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

# --- Main Data Query and Ranking Calculation ---
main_query = """
WITH game_stats AS (
    SELECT
        g.game_name,
        COUNT(r.score) as n,
        AVG(r.score) as x_bar,
        (SELECT COUNT(*) FROM critics) AS total_critics
    FROM games g
    LEFT JOIN ratings r ON g.id = r.game_id
    WHERE g.upcoming IS FALSE AND r.score IS NOT NULL
    GROUP BY g.game_name
), global_stats AS (
    SELECT 2 AS C, (SELECT AVG(score) FROM ratings WHERE score IS NOT NULL) as m
)
SELECT
    gs.game_name,
    gs.x_bar as average_score,
    gs.n as number_of_ratings,
    ( (gs.n * gs.x_bar) + (glob.C * glob.m) ) / ( gs.n + glob.C ) AS final_adjusted_score
FROM game_stats gs, global_stats glob;
"""
rankings_df = conn.query(main_query)

# Add Rank columns based on scores
rankings_df = rankings_df.sort_values("final_adjusted_score", ascending=False).reset_index(drop=True)
rankings_df['Rank'] = rankings_df.index + 1
rankings_df['Unadjusted Rank'] = rankings_df['average_score'].rank(method='min', ascending=False).astype(int)

# --- Top & Bottom Ranked Games Showcase ---
with st.container(border=True):
    st.subheader("Top & Bottom Ranked Games")
    
    # Get top and bottom ranked games
    best_adjusted = rankings_df.loc[rankings_df['Rank'].idxmin()]
    worst_adjusted = rankings_df.loc[rankings_df['Rank'].idxmax()]
    best_unadjusted = rankings_df.loc[rankings_df['Unadjusted Rank'].idxmin()]
    worst_unadjusted = rankings_df.loc[rankings_df['Unadjusted Rank'].idxmax()]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Final Ranking")
        st.markdown(f"**#1**: {best_adjusted['game_name']}")
        st.markdown(f"**Last**: {worst_adjusted['game_name']} (#{worst_adjusted['Rank']})")
    with col2:
        st.markdown("##### Unadjusted Ranking")
        st.markdown(f"**#1**: {best_unadjusted['game_name']}")
        st.markdown(f"**Last**: {worst_unadjusted['game_name']} (#{worst_unadjusted['Unadjusted Rank']})")


# --- Tabs for Full Rankings and Upcoming Games ---
tab1, tab2 = st.tabs(["Full Rankings", "Upcoming Games"])

with tab1:
    st.dataframe(
        rankings_df[['Rank', 'game_name', 'Unadjusted Rank', 'number_of_ratings']],
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "game_name": "Game",
            "Unadjusted Rank": "Unadjusted Rank",
            "number_of_ratings": "Ratings"
        },
        hide_index=True,
        use_container_width=True
    )

with tab2:
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
        st.dataframe(
            upcoming_games_df.rename(columns={"game_name": "Game", "nominated_by": "Nominated By"}),
            hide_index=True,
            use_container_width=True
        )

# --- Log Out Button ---
if st.button("Log out"):
    st.session_state["password_correct"] = False
    st.rerun()