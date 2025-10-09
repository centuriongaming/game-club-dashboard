# pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
st.title("Dashboard & Leaderboards")

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

# --- Main Data Query and Game Ranking Calculation ---
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
rankings_df = rankings_df.sort_values("final_adjusted_score", ascending=False).reset_index(drop=True)
rankings_df['Rank'] = rankings_df.index + 1
rankings_df['Unadjusted Rank'] = rankings_df['average_score'].rank(method='min', ascending=False).astype(int)

# --- Top & Bottom Ranked Games Showcase ---
with st.container(border=True):
    st.subheader("Top & Bottom Ranked Games")
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


# --- Tabs for Main Content ---
tab1, tab2, tab3 = st.tabs(["Game Rankings", "Critic Analysis", "Upcoming Games"])

# --- Game Rankings Tab ---
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

# --- Critic Analysis Tab ---
with tab2:
    st.subheader("Critics by Participation")
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
    critic_participation_df['participation_rate'] = (critic_participation_df['ratings_given'] / critic_participation_df['total_games']) * 100
    st.dataframe(
        critic_participation_df[['critic_name', 'ratings_given', 'participation_rate', 'average_score']],
        column_config={
            "critic_name": "Critic",
            "ratings_given": "Ratings Given",
            "average_score": st.column_config.ProgressColumn("Average Score",format="%.2f",min_value=0,max_value=10),
            "participation_rate": st.column_config.ProgressColumn("Participation",format="%.1f%%",min_value=0,max_value=100)
        },
        hide_index=True, use_container_width=True
    )

    st.subheader("Visual Breakdowns")
    col1, col2 = st.columns(2)
    with col1:
        nomination_query = """
            SELECT c.critic_name, COUNT(g.id) AS nomination_count
            FROM games g JOIN critics c ON g.nominated_by = c.id
            GROUP BY c.critic_name ORDER BY nomination_count DESC;
        """
        nomination_df = conn.query(nomination_query)
        fig = go.Figure(data=[go.Pie(labels=nomination_df['critic_name'], values=nomination_df['nomination_count'], hole=.3, textinfo='label+percent')])
        fig.update_layout(title_text="Nominations by Critic", showlegend=False, height=350, margin=dict(l=1, r=1, t=30, b=1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        binned_ratings_query = """
            SELECT c.critic_name, FLOOR(r.score) as score_bin, COUNT(r.id) as rating_count
            FROM ratings r JOIN critics c ON r.critic_id = c.id
            WHERE r.score IS NOT NULL GROUP BY c.critic_name, score_bin;
        """
        binned_df = conn.query(binned_ratings_query)
        if not binned_df.empty:
            binned_pivot = binned_df.pivot(index='score_bin', columns='critic_name', values='rating_count').fillna(0)
            st.bar_chart(binned_pivot, height=350)
            st.caption("Score Distribution by Critic")

    st.subheader("Controversial Critic Analysis")
    controversy_query = """
        WITH game_avg AS (SELECT game_id, AVG(score) as avg_game_score FROM ratings WHERE score IS NOT NULL GROUP BY game_id)
        SELECT c.critic_name, AVG(ABS(r.score - ga.avg_game_score)) as controversy_score
        FROM ratings r JOIN critics c ON r.critic_id = c.id JOIN game_avg ga ON r.game_id = ga.game_id
        WHERE r.score IS NOT NULL GROUP BY c.critic_name ORDER BY controversy_score DESC;
    """
    critic_controversy_df = conn.query(controversy_query)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Most Contrarian")
        st.dataframe(critic_controversy_df.head(5), hide_index=True)
    with col2:
        st.markdown("##### Least Contrarian")
        st.dataframe(critic_controversy_df.sort_values("controversy_score", ascending=True).head(5), hide_index=True)

# --- Upcoming Games Tab ---
with tab3:
    upcoming_games_query = """
        SELECT g.game_name, c.critic_name AS nominated_by
        FROM games g LEFT JOIN critics c ON g.nominated_by = c.id
        WHERE g.upcoming IS TRUE ORDER BY g.game_name ASC;
    """
    upcoming_games_df = conn.query(upcoming_games_query)
    if upcoming_games_df.empty:
        st.info("There are currently no games marked as upcoming.")
    else:
        st.dataframe(
            upcoming_games_df.rename(columns={"game_name": "Game", "nominated_by": "Nominated By"}),
            hide_index=True, use_container_width=True
        )

# --- Log Out Button ---
if st.button("Log out"):
    st.session_state["password_correct"] = False
    st.rerun()