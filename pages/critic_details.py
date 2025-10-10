# pages/critic_details.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import aliased
from utils import check_auth, get_sqla_session, load_queries
from database_models import Critic, Game, Rating

# --- Initial Setup ---
check_auth()
session = get_sqla_session()
queries = load_queries()

# --- Page Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")
st.title("Critic Deep Dive")

# --- Critic Selection ---
critics = session.query(Critic.id, Critic.critic_name).order_by(Critic.critic_name).all()
critic_map = {name: id for id, name in critics}
selected_critic_name = st.selectbox("Select a Critic to Analyze:", critic_map.keys())

if selected_critic_name:
    selected_critic_id = critic_map[selected_critic_name]

    # --- Data Fetching ---
    game_stats_subq = (
        sa.select(
            Game.id.label("game_id"), Game.game_name,
            sa.func.avg(Rating.score).label("avg_game_score"),
            (sa.func.count(Rating.critic_id).cast(sa.Float) / sa.select(sa.func.count(Critic.id)).scalar_subquery()).label("participation_rate")
        )
        .join(Rating, Game.id == Rating.game_id, isouter=True).where(Game.upcoming == False)
        .group_by(Game.id, Game.game_name).subquery()
    )
    
    critic_rating_alias = aliased(Rating)
    details_stmt = (
        sa.select(
            game_stats_subq.c.game_name, game_stats_subq.c.avg_game_score,
            game_stats_subq.c.participation_rate, critic_rating_alias.score.label("critic_score")
        )
        .select_from(game_stats_subq)
        .join(critic_rating_alias, (game_stats_subq.c.game_id == critic_rating_alias.game_id) & (critic_rating_alias.critic_id == selected_critic_id), isouter=True)
    )
    details_df = pd.read_sql(details_stmt, session.bind)
    
    total_nominations = session.query(sa.func.count(Game.id)).filter(Game.nominated_by == selected_critic_id).scalar()
    
    # FIX: Use the correct key from queries.json
    controversy_breakdown_df = pd.read_sql(sa.text(queries['get_critic_controversy_breakdown']), session.bind)
    
    # Check if critic exists in the breakdown dataframe
    if selected_critic_name in controversy_breakdown_df['critic_name'].values:
        critic_breakdown = controversy_breakdown_df[controversy_breakdown_df['critic_name'] == selected_critic_name].iloc[0]
        final_controversy_score = critic_breakdown['controversy_score']
    else:
        # Handle case for critics with no ratings (they won't be in the controversy query result)
        critic_breakdown = {'observed_score': 0, 'n': 0, 'prior_score': controversy_breakdown_df['prior_score'].mean(), 'credibility_constant': 15, 'controversy_score': controversy_breakdown_df['prior_score'].mean()}
        final_controversy_score = critic_breakdown['prior_score']


    # --- Calculations & Scorecard ---
    critic_ratings = details_df.dropna(subset=['critic_score'])
    n = len(critic_ratings)
    
    participation_rate = (n / len(details_df)) * 100 if len(details_df) > 0 else 0
    avg_score = critic_ratings['critic_score'].mean() if not critic_ratings.empty else 0

    st.subheader(f"Scorecard for {selected_critic_name}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participation Rate", f"{participation_rate:.1f}%")
        col2.metric("Average Score Given", f"{avg_score:.2f}")
        col3.metric("Final Controversy Score", f"{final_controversy_score:.3f}")
        col4.metric("Games Nominated", f"{total_nominations}")

    # --- Controversy Score Breakdown Expander ---
    with st.expander("**Controversy Score Breakdown**"):
        observed_score = critic_breakdown['observed_score']
        n_calc = critic_breakdown['n'] # Use n from the query for the breakdown display
        prior_score = critic_breakdown['prior_score']
        C = critic_breakdown['credibility_constant']
        credibility_weight = n_calc / (n_calc + C) if (n_calc + C) > 0 else 0

        st.markdown("The final score is a weighted average of the critic's observed score and the group's average, adjusted for the number of games rated (`n`).")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Observed Score", f"{observed_score:.3f}", help="The critic's raw, un-adjusted controversy score.")
        col2.metric("2. Games Rated (n)", f"{n_calc}", help="The number of games rated by the critic, used to determine credibility.")
        col3.metric("3. Group Average", f"{prior_score:.3f}", help="The average controversy score for the entire group.")
        col4.metric("Credibility Weight", f"{credibility_weight:.1%}", help=f"The weight given to the observed score. Calculated as n / (n + C), where C is {C}.")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = (\text{Weight} \times \text{Observed}) + (1 - \text{Weight}) \times \text{Group Average} $$')
        calculation_str = f"= ({credibility_weight:.2f} * {observed_score:.3f}) + ({1-credibility_weight:.2f} * {prior_score:.3f}) = **{final_controversy_score:.3f}**"
        st.markdown(calculation_str)

    # --- Detailed Analysis Tabs ---
    st.subheader("Detailed Analysis")
    
    critic_ratings['deviation'] = (critic_ratings['critic_score'] - critic_ratings['avg_game_score']).abs()
    most_contrarian_ratings = critic_ratings.sort_values('deviation', ascending=False).head(10)
    details_df['play_decision_diff'] = (details_df['critic_score'].notna().astype(int) - details_df['participation_rate']).abs()
    most_contrarian_plays = details_df.sort_values('play_decision_diff', ascending=False).head(10)
    most_contrarian_plays['participation_rate_percent']