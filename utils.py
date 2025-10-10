# utils.py
import json
import streamlit as st
import pandas as pd
import sqlalchemy as sa
import streamlit as st
from database_models import Critic, Game, Rating

def check_auth():
    """Checks if the user is authenticated."""
    if not st.session_state.get("password_correct", False):
        st.error("🔒 You need to log in first.")
        st.stop()

@st.cache_resource
def get_sqla_session():
    """Returns a cached SQLAlchemy session object."""
    return st.connection("mydb", type="sql").session

@st.cache_data
def load_queries():
    """Loads and joins raw SQL queries from the JSON file."""
    with open('queries.json', 'r') as f:
        queries = json.load(f)
    for key, value in queries.items():
        if isinstance(value, list):
            queries[key] = " ".join(value)
    return queries

@st.cache_data
def calculate_controversy_scores(_session):
    """
    Fetches all required data and performs the full, multi-step controversy
    calculation in Pandas, returning a dataframe with the final scores.
    """
    # 1. Fetch Base Data
    critics_df = pd.read_sql(sa.select(Critic.id, Critic.critic_name), _session.bind)
    all_games_df = pd.read_sql(sa.select(Game.id.label("game_id"), Game.game_name).where(Game.upcoming == False), _session.bind)
    all_ratings_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), _session.bind)
    
    # 2. Calculate Game Stats
    game_stats = all_ratings_df.groupby('game_id')['score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_game_score', 'count': 'rating_count'})
    game_stats['participation_rate'] = game_stats['rating_count'] / len(critics_df)
    game_stats = pd.merge(all_games_df, game_stats, on='game_id', how='left').fillna({'avg_game_score': 0, 'participation_rate': 0})
    
    # 3. Calculate Critic Stats (n)
    critic_stats = all_ratings_df.groupby('critic_id').size().reset_index(name='n')

    # 4. Build a "Scaffold" of every critic-game pair
    scaffold = critics_df.merge(all_games_df, how='cross')
    scaffold = pd.merge(scaffold, game_stats.drop(columns=['game_name']), on='game_id', how='left')
    scaffold = pd.merge(scaffold, critic_stats, left_on='id', right_on='critic_id', how='left')
    scaffold = pd.merge(scaffold, all_ratings_df, on=['critic_id', 'game_id'], how='left')
    scaffold['n'] = scaffold['n'].fillna(0).astype(int)

    # 5. Calculate Deviations
    scaffold['normalized_score_deviation'] = (scaffold['score'] - scaffold['avg_game_score']).abs() / 10.0
    
    def calculate_play_deviation(row):
        if row['n'] < 10: return 0
        is_rated = pd.notna(row['score'])
        if is_rated and row['participation_rate'] <= 0.5: return 1.0 - row['participation_rate']
        elif not is_rated and row['participation_rate'] > 0.5: return row['participation_rate']
        return 0
    scaffold['play_deviation'] = scaffold.apply(calculate_play_deviation, axis=1)

    # 7. Calculate Observed Score
    scaffold['total_deviation'] = (0.5 * scaffold['normalized_score_deviation'].fillna(0)) + (0.5 * scaffold['play_deviation'])
    observed_df = scaffold.groupby(['critic_name', 'n'])['total_deviation'].mean().reset_index().rename(columns={'total_deviation': 'observed_score'})

    # 8. Apply Bayesian Shrinkage
    prior_score = observed_df['observed_score'].mean()
    C = 15
    observed_df['credibility_weight'] = observed_df['n'] / (observed_df['n'] + C)
    observed_df['final_controversy_score'] = (observed_df['credibility_weight'] * observed_df['observed_score']) + ((1 - observed_df['credibility_weight']) * prior_score)
    
    # Add prior_score and C for breakdown display later
    observed_df['prior_score'] = prior_score
    observed_df['credibility_constant'] = C
    
    return observed_df.sort_values('final_controversy_score', ascending=False)
