# pages/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlalchemy as sa
from utils import check_auth, get_sqla_session, load_queries
from database_models import Critic, Game, Rating

# --- Initial Setup ---
check_auth()
session = get_sqla_session()
queries = load_queries() # For the one remaining raw SQL query

# --- Page Configuration ---
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Dashboard & Leaderboards")

# --- Consistent Critic Color Map ---
critics_list = session.query(Critic.critic_name).order_by(Critic.critic_name).all()
critic_names = [c[0] for c in critics_list]
colors = px.colors.qualitative.Plotly
color_map = {name: colors[i % len(colors)] for i, name in enumerate(critic_names)}

# --- Key Performance Indicators ---
with st.container(border=True):
    total_ratings = session.query(sa.func.count(Rating.id)).scalar()
    avg_score = session.query(sa.func.avg(Rating.score)).scalar()
    
    # For participation, we need total critics and total rated games
    total_critics = session.query(sa.func.count(Critic.id)).scalar()
    total_games = session.query(sa.func.count(Game.id)).where(Game.upcoming == False).scalar()
    participation_rate = (total_ratings / (total_critics * total_games)) * 100 if (total_critics * total_games) > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ratings Given", f"{total_ratings}")
    col2.metric("Overall Average Score", f"{avg_score:.2f}" if avg_score else "N/A")
    col3.metric("Group Participation", f"{participation_rate:.1f}%")

# --- Game Ranking Calculation with SQLAlchemy ---
# Subquery to get stats for each game (equivalent to the game_stats CTE)
game_stats_subq = (
    sa.select(
        Game.game_name,
        sa.func.count(Rating.score).label("n"),
        sa.func.avg(Rating.score).label("x_bar")
    )
    .join(Rating, Game.id == Rating.game_id)
    .where(Game.upcoming == False, Rating.score.is_not(None))
    .group_by(Game.game_name)
    .subquery()
)

# Subquery for global stats (equivalent to the global_stats CTE)
global_stats_subq = (
    sa.select(
        sa.literal(2).label("C"),
        session.query(sa.func.avg(Rating.score)).scalar_subquery().label("m")
    ).subquery()
)

# Main query to calculate final rankings
rankings_stmt = (
    sa.select(
        game_stats_subq.c.game_name,
        game_stats_subq.c.x_bar.label("average_score"),
        game_stats_subq.c.n.label("number_of_ratings"),
        (
            (game_stats_subq.c.n * game_stats_subq.c.x_bar + global_stats_subq.c.C * global_stats_subq.c.m) /
            (game_stats_subq.c.n + global_stats_subq.c.C)
        ).label("final_adjusted_score")
    )
    .select_from(game_stats_subq)
    .cross_join(global_stats_subq)
)

rankings_df = pd.read_sql(rankings_stmt, session.bind)
rankings_df = rankings_df.sort_values("final_adjusted_score", ascending=False).reset_index(drop=True)
rankings_df['Rank'] = rankings_df.index + 1
rankings_df['Unadjusted Rank'] = rankings_df['average_score'].rank(method='min', ascending=False).astype(int)

# --- Top & Bottom Ranked Games Showcase ---
with st.container(border=True):
    st.subheader("Top & Bottom Ranked Games")
    # This logic remains the same as it operates on the dataframe
    best_adjusted = rankings_df.loc[rankings_df['Rank'].idxmin()]
    worst_adjusted = rankings_df.loc[rankings_df['Rank'].idxmax()]
    best_unadjusted = rankings_df.loc[rankings_df['Unadjusted Rank'].idxmin()]
    worst_unadjusted = rankings_df.loc[rankings_df['Unadjusted Rank'].idxmax()]
    # Display logic here... (same as before)

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
        hide_index=True, use_container_width=True
    )

# --- Critic Analysis Tab ---
with tab2:
    st.subheader("Visual Breakdowns")
    col1, col2 = st.columns(2)
    with col1:
        # Nominations Query
        nomination_stmt = (
            sa.select(Critic.critic_name, sa.func.count(Game.id).label("nomination_count"))
            .join(Game, Critic.id == Game.nominated_by)
            .group_by(Critic.critic_name)
        )
        nomination_df = pd.read_sql(nomination_stmt, session.bind)
        # Display logic here... (same as before)

    with col2:
        # Score Distribution Query
        binned_stmt = (
            sa.select(
                Critic.critic_name,
                sa.func.floor(Rating.score).label("score_bin"),
                sa.func.count(Rating.id).label("rating_count")
            )
            .join(Rating, Critic.id == Rating.critic_id)
            .where(Rating.score.is_not(None))
            .group_by(Critic.critic_name, "score_bin")
        )
        binned_df = pd.read_sql(binned_stmt, session.bind)
        # Display logic here... (same as before)

    st.subheader("Critics by Participation")
    participation_stmt = (
        sa.select(
            Critic.critic_name,
            sa.func.count(Rating.score).label("ratings_given"),
            sa.func.avg(Rating.score).label("average_score")
        )
        .join(Rating, Critic.id == Rating.critic_id, isouter=True)
        .group_by(Critic.critic_name)
        .order_by(sa.desc("ratings_given"))
    )
    critic_participation_df = pd.read_sql(participation_stmt, session.bind)
    critic_participation_df['participation_rate'] = (critic_participation_df['ratings_given'] / total_games) * 100 if total_games > 0 else 0
    # Display logic here... (same as before)
    
    st.subheader("Critic Controversy Ranking")
    st.caption("The score represents how a critic's rating and participation deviates from the group consensus.")
    critic_controversy_df = pd.read_sql(queries['get_critic_controversy'], session.bind)
    critic_controversy_df['Rank'] = critic_controversy_df['controversy_score'].rank(method='min', ascending=False).astype(int)
    # Display logic here... (same as before)

# --- Upcoming Games Tab ---
with tab3:
    upcoming_stmt = (
        sa.select(Game.game_name, Critic.critic_name.label("nominated_by"))
        .join(Critic, Game.nominated_by == Critic.id, isouter=True)
        .where(Game.upcoming == True)
        .order_by(Game.game_name)
    )
    upcoming_games_df = pd.read_sql(upcoming_stmt, session.bind)
    # Display logic here... (same as before)

# ... (rest of display logic and logout button)