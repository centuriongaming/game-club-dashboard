# pages/critic_details.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating

# --- Initial Setup ---
check_auth()
session = get_sqla_session()

# --- Page Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")
st.title("Critic Deep Dive")

# --- Critic Selection ---
critics_df = pd.read_sql(sa.select(Critic.id, Critic.critic_name).order_by(Critic.critic_name), session.bind)
critic_map = pd.Series(critics_df['id'].values, index=critics_df['critic_name']).to_dict()
selected_critic_name = st.selectbox("Select a Critic to Analyze:", critic_map.keys())

if selected_critic_name:
    selected_critic_id = critic_map[selected_critic_name]

    # --- Full Controversy Calculation in Python/Pandas ---
    
    # 1. Fetch Base Data
    all_games_df = pd.read_sql(sa.select(Game.id.label("game_id"), Game.game_name).where(Game.upcoming == False), session.bind)
    all_ratings_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), session.bind)
    
    # 2. Calculate Game Stats
    game_stats = all_ratings_df.groupby('game_id')['score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_game_score', 'count': 'rating_count'})
    game_stats['participation_rate'] = game_stats['rating_count'] / len(critics_df)
    game_stats = pd.merge(all_games_df, game_stats, on='game_id', how='left').fillna({'avg_game_score': 0, 'participation_rate': 0})
    
    # 3. Calculate Critic Stats (n)
    critic_stats = all_ratings_df.groupby('critic_id').size().reset_index(name='n')

    # 4. Build a "Scaffold" of every critic-game pair
    scaffold = critics_df.merge(all_games_df, how='cross')
    
    # FIX: Drop the redundant 'game_name' column from game_stats before merging to prevent a name collision
    scaffold = pd.merge(scaffold, game_stats.drop(columns=['game_name']), on='game_id', how='left')
    
    scaffold = pd.merge(scaffold, critic_stats, left_on='id', right_on='critic_id', how='left')
    scaffold = pd.merge(scaffold, all_ratings_df, on=['critic_id', 'game_id'], how='left')
    scaffold['n'] = scaffold['n'].fillna(0).astype(int)

    # 5. Calculate Deviations
    scaffold['normalized_score_deviation'] = (scaffold['score'] - scaffold['avg_game_score']).abs() / 10.0
    
    def calculate_play_deviation(row):
        if row['n'] < 10:
            return 0
        is_rated = pd.notna(row['score'])
        if is_rated and row['participation_rate'] <= 0.5:
            return 1.0 - row['participation_rate']
        elif not is_rated and row['participation_rate'] > 0.5:
            return row['participation_rate']
        return 0
    scaffold['play_deviation'] = scaffold.apply(calculate_play_deviation, axis=1)

    # 7. Calculate Observed Score for each critic
    scaffold['total_deviation'] = (0.5 * scaffold['normalized_score_deviation'].fillna(0)) + (0.5 * scaffold['play_deviation'])
    observed_controversy_df = scaffold.groupby(['critic_name', 'n'])['total_deviation'].mean().reset_index().rename(columns={'total_deviation': 'observed_score'})

    # 8. Apply Bayesian Shrinkage
    prior_score = observed_controversy_df['observed_score'].mean()
    C = 15
    observed_controversy_df['credibility_weight'] = observed_controversy_df['n'] / (observed_controversy_df['n'] + C)
    observed_controversy_df['final_controversy_score'] = (observed_controversy_df['credibility_weight'] * observed_controversy_df['observed_score']) + ((1 - observed_controversy_df['credibility_weight']) * prior_score)
    
    # --- Data for Display ---
    critic_breakdown = observed_controversy_df.loc[observed_controversy_df['critic_name'] == selected_critic_name].iloc[0]
    details_df = scaffold.loc[scaffold['critic_name'] == selected_critic_name]
    critic_ratings = details_df.dropna(subset=['score'])
    total_nominations = session.query(sa.func.count(Game.id)).filter(Game.nominated_by == selected_critic_id).scalar()
    
    # --- Scorecard ---
    participation_rate = (critic_breakdown['n'] / len(all_games_df)) * 100 if len(all_games_df) > 0 else 0
    avg_score = critic_ratings['score'].mean() if not critic_ratings.empty else 0

    st.subheader(f"Scorecard for {selected_critic_name}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participation Rate", f"{participation_rate:.1f}%")
        col2.metric("Average Score Given", f"{avg_score:.2f}")
        col3.metric("Final Controversy Score", f"{critic_breakdown['final_controversy_score']:.3f}")
        col4.metric("Games Nominated", f"{total_nominations}")

    # (Expander and Tabs code remains the same)
    with st.expander("**Controversy Score Breakdown**"):
        st.markdown("The final score is a weighted average of the critic's observed score and the group's average, adjusted for the number of games rated (`n`).")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Observed Score", f"{critic_breakdown['observed_score']:.3f}", help="The critic's raw, un-adjusted controversy score.")
        col2.metric("2. Games Rated (n)", f"{critic_breakdown['n']:.0f}", help="The number of games rated by the critic, used to determine credibility.")
        col3.metric("3. Group Average", f"{prior_score:.3f}", help="The average controversy score for the entire group.")
        col4.metric("Credibility Weight", f"{critic_breakdown['credibility_weight']:.1%}", help=f"The weight given to the observed score. Calculated as n / (n + C), where C is {C}.")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = (\text{Weight} \times \text{Observed}) + (1 - \text{Weight}) \times \text{Group Average} $$')
        calculation_str = f"= ({critic_breakdown['credibility_weight']:.2f} * {critic_breakdown['observed_score']:.3f}) + ({1-critic_breakdown['credibility_weight']:.2f} * {prior_score:.3f}) = **{critic_breakdown['final_controversy_score']:.3f}**"
        st.markdown(calculation_str)
        st.subheader("Detailed Analysis")
        most_contrarian_ratings = critic_ratings.sort_values('normalized_score_deviation', ascending=False).head(10)
        most_contrarian_plays = details_df.sort_values('play_deviation', ascending=False).head(10)
        most_contrarian_plays['participation_rate_percent'] = most_contrarian_plays['participation_rate'] * 100
    
    tab1, tab2, tab3 = st.tabs(["Most Contrarian Ratings", "Contrarian Participation", "Full Rating History"])
    
    with tab1:
        st.dataframe(
            most_contrarian_ratings.rename(columns={'score': 'critic_score', 'normalized_score_deviation': 'deviation'})[['game_name', 'critic_score', 'avg_game_score', 'deviation']],
            column_config={
                "game_name": "Game", "critic_score": "Their Score",
                "avg_game_score": "Group Average", "deviation": st.column_config.NumberColumn("Score Deviation (0-1)", format="%.3f")
            },
            hide_index=True, width='stretch'
        )

    with tab2:
        st.dataframe(
            most_contrarian_plays.rename(columns={'score': 'critic_score'})[['game_name', 'critic_score', 'participation_rate_percent']],
            column_config={
                "game_name": "Game", "critic_score": "Their Score (if played)",
                "participation_rate_percent": st.column_config.ProgressColumn("Group Participation Rate", format="%d%%", min_value=0, max_value=100)
            },
            hide_index=True, width='stretch'
        )

    with tab3:
        st.dataframe(
            critic_ratings.rename(columns={'score': 'critic_score'})[['game_name', 'critic_score', 'avg_game_score']],
            column_config={
                "game_name": "Game", "critic_score": "Their Score", "avg_game_score": "Group Average"
            },
            hide_index=True, width='stretch'
        )