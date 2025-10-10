# pages/critic_details.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import aliased
from utils import check_auth, get_sqla_session, load_queries
from models import Critic, Game, Rating

# --- Initial Setup ---
check_auth()
session = get_sqla_session()
queries = load_queries()

# --- Page Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")
st.title("Critic Deep Dive")

# --- Critic Selection (using SQLAlchemy) ---
critics = session.query(Critic.id, Critic.critic_name).order_by(Critic.critic_name).all()
critic_map = {name: id for id, name in critics}
selected_critic_name = st.selectbox("Select a Critic to Analyze:", critic_map.keys())

if selected_critic_name:
    selected_critic_id = critic_map[selected_critic_name]

    # --- Data Fetching with SQLAlchemy ---
    
    # Create subquery for game stats (avg score, participation_rate)
    game_stats_subq = (
        sa.select(
            Game.id.label("game_id"),
            Game.game_name,
            sa.func.avg(Rating.score).label("avg_game_score"),
            (sa.func.count(Rating.critic_id).cast(sa.Float) / session.query(sa.func.count(Critic.id)).scalar_one()).label("participation_rate")
        )
        .join(Rating, Game.id == Rating.game_id, isouter=True)
        .where(Game.upcoming == False)
        .group_by(Game.id, Game.game_name)
        .subquery()
    )
    
    # Main query to get details, joining game stats with the selected critic's ratings
    critic_rating_alias = aliased(Rating)
    details_stmt = (
        sa.select(
            game_stats_subq.c.game_name,
            game_stats_subq.c.avg_game_score,
            game_stats_subq.c.participation_rate,
            critic_rating_alias.score.label("critic_score")
        )
        .select_from(game_stats_subq)
        .join(critic_rating_alias, (game_stats_subq.c.game_id == critic_rating_alias.game_id) & (critic_rating_alias.critic_id == selected_critic_id), isouter=True)
    )
    details_df = pd.read_sql(details_stmt, session.bind)
    
    # Fetch simple count of nominations
    total_nominations = session.query(sa.func.count(Game.id)).filter(Game.nominated_by == selected_critic_id).scalar()
    
    # Fetch the controversy breakdown using the remaining raw SQL query
    controversy_breakdown_df = pd.read_sql(sa.text(queries['get_critic_controversy_breakdown']), session.bind)
    critic_breakdown = controversy_breakdown_df[controversy_breakdown_df['critic_name'] == selected_critic_name].iloc[0]

    # --- Calculations & Scorecard ---
    critic_ratings = details_df.dropna(subset=['critic_score'])
    participation_rate = (len(critic_ratings) / len(details_df)) * 100 if len(details_df) > 0 else 0
    avg_score = critic_ratings['critic_score'].mean() if not critic_ratings.empty else 0

    st.subheader(f"Scorecard for {selected_critic_name}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participation Rate", f"{participation_rate:.1f}%")
        col2.metric("Average Score Given", f"{avg_score:.2f}")
        col3.metric("Final Controversy Score", f"{critic_breakdown['controversy_score']:.3f}")
        col4.metric("Games Nominated", f"{total_nominations}")

    # --- Controversy Score Breakdown Expander ---
    with st.expander("**Controversy Score Breakdown**"):
        observed_score = critic_breakdown['observed_score']
        n = critic_breakdown['n']
        prior_score = critic_breakdown['prior_score']
        C = critic_breakdown['credibility_constant']
        credibility_weight = n / (n + C) if (n + C) > 0 else 0

        st.markdown("The final score is a weighted average of the critic's observed score and the group's average, adjusted for the number of games rated (`n`).")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Observed Score", f"{observed_score:.3f}", help="The critic's raw, un-adjusted controversy score.")
        col2.metric("2. Games Rated (n)", f"{n}", help="The number of games rated by the critic, used to determine credibility.")
        col3.metric("3. Group Average", f"{prior_score:.3f}", help="The