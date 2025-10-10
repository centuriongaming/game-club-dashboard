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

# utils.py
@st.cache_data
def calculate_controversy_scores(_session):
    """
    Performs the full, multi-step controversy calculation for ALL critics
    and returns the final scores and the detailed scaffold dataframe.
    This is the single source of truth for controversy scores.
    """
    # 1. Fetch Base Data
    critics_df = pd.read_sql(sa.select(Critic.id, Critic.critic_name), _session.bind)
    all_games_df = pd.read_sql(sa.select(Game.id.label("game_id"), Game.game_name).where(Game.upcoming == False), _session.bind)
    all_ratings_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), _session.bind)

    if all_ratings_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Calculate Game Stats
    global_avg_score = all_ratings_df['score'].mean()
    game_stats = all_ratings_df.groupby('game_id')['score'].agg(['mean', 'count']).rename(columns={'mean': 'avg_game_score', 'count': 'rating_count'})
    game_stats['participation_rate'] = game_stats['rating_count'] / len(critics_df)
    game_stats = pd.merge(all_games_df, game_stats, on='game_id', how='left')
    game_stats['avg_game_score'] = game_stats['avg_game_score'].fillna(global_avg_score)
    game_stats = game_stats.fillna({'rating_count': 0, 'participation_rate': 0})
    
    # 3. Build the Scaffold (we no longer need a separate critic_stats calculation)
    scaffold = critics_df.merge(all_games_df, how='cross')
    scaffold = pd.merge(scaffold, game_stats.drop(columns=['game_name']), on='game_id', how='left')
    scaffold = pd.merge(scaffold, all_ratings_df, on=['critic_id', 'game_id'], how='left')

    # 4. Calculate Deviations
    scaffold['normalized_score_deviation'] = (scaffold['score'] - scaffold['avg_game_score']).abs() / 10.0
    
    # We need a temporary 'n' on the scaffold for the play_deviation calculation
    n_temp = scaffold.dropna(subset=['score']).groupby('critic_id').size().reset_index(name='n')
    scaffold = pd.merge(scaffold, n_temp, on='critic_id', how='left')
    scaffold['n'] = scaffold['n'].fillna(0).astype(int)

    def calculate_play_deviation(row):
        if row['n'] < 10: return 0
        is_rated = pd.notna(row['score'])
        if is_rated and row['participation_rate'] <= 0.5: return 1.0 - row['participation_rate']
        elif not is_rated and row['participation_rate'] > 0.5: return row['participation_rate']
        return 0
    scaffold['play_deviation'] = scaffold.apply(calculate_play_deviation, axis=1)

    # 5. Calculate Observed Score
    rated_scaffold = scaffold.dropna(subset=['score']).copy()
    avg_score_dev = rated_scaffold.groupby('critic_id')['normalized_score_deviation'].mean().reset_index(name='avg_score_deviation')
    avg_play_dev = scaffold.groupby('critic_id')['play_deviation'].mean().reset_index(name='avg_play_deviation')
    
    # --- THIS SECTION CONTAINS THE FIX ---
    # 6. Re-calculate 'n' from the scaffold to guarantee it matches the details page.
    # This is now the definitive count of games rated.
    n_definitive = scaffold.dropna(subset=['score']).groupby('critic_id').size().reset_index(name='n')

    observed_df = pd.merge(critics_df, n_definitive, left_on='id', right_on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_score_dev, on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_play_dev, on='critic_id', how='left')
    observed_df = observed_df.fillna({'n': 0, 'avg_score_deviation': 0, 'avg_play_deviation': 0})
    observed_df['n'] = observed_df['n'].astype(int)
    observed_df['observed_score'] = (0.5 * observed_df['avg_score_deviation']) + (0.5 * observed_df['avg_play_deviation'])
    
    # 7. Apply Bayesian Shrinkage using the definitive 'n'
    prior_score = observed_df['observed_score'].mean()
    total_games = len(all_games_df)
    credibility_threshold = max(1, total_games // 2)

    observed_df['credibility_weight'] = observed_df['n'] / (observed_df['n'] + credibility_threshold)
    observed_df['final_controversy_score'] = (observed_df['credibility_weight'] * observed_df['observed_score']) + ((1 - observed_df['credibility_weight']) * prior_score)
    observed_df['prior_score'] = prior_score
    observed_df['credibility_threshold'] = credibility_threshold

    return observed_df, scaffold

def calculate_custom_game_rankings(games_df, critics_df, ratings_df):
    """
    Calculates game rankings AND detailed critic stats.
    
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The final ranked DataFrame.
            - pd.DataFrame: The critics DataFrame with personal avg and std dev.
    """
    # 1. Pre-calculate personal stats for every critic
    critic_stats = ratings_df.groupby('critic_id')['score'].agg(['mean', 'std']).rename(columns={'mean': 'critic_avg', 'std': 'critic_std'})
    critic_stats['critic_std'] = critic_stats['critic_std'].fillna(0)
    critics_with_stats_df = pd.merge(critics_df, critic_stats, left_on='id', right_on='critic_id', how='left')
    global_avg_fallback = ratings_df['score'].mean()

    def calculate_game_rank(game_group):
        n = len(game_group)
        game_avg = game_group['score'].mean()
        
        raters_ids = set(game_group['critic_id'])
        all_critic_ids = set(critics_with_stats_df['id'])
        skipper_ids = all_critic_ids - raters_ids
        n_skipped = len(skipper_ids)
        
        if n_skipped > 0:
            skipper_stats = critics_with_stats_df[critics_with_stats_df['id'].isin(skipper_ids)]
            skipper_stats['pessimistic_score'] = skipper_stats['critic_avg'] - skipper_stats['critic_std']
            pessimistic_prior = skipper_stats['pessimistic_score'].mean()
        else:
            pessimistic_prior = global_avg_fallback
        
        if pd.isna(pessimistic_prior):
            pessimistic_prior = global_avg_fallback

        numerator = (n * game_avg) + (n_skipped * pessimistic_prior)
        denominator = n + n_skipped
        adjusted_score = numerator / denominator if denominator > 0 else 0
        
        # UPDATE: Return all the components we need for the breakdown
        return pd.Series({
            'number_of_ratings': n,
            'average_score': game_avg,
            'pessimistic_prior': pessimistic_prior,
            'number_of_skippers': n_skipped,
            'final_adjusted_score': adjusted_score
        })

    # ... (the rest of the function is the same)
    rankings_df = ratings_df.groupby('game_id').apply(calculate_game_rank)
    rankings_df = pd.merge(games_df, rankings_df, left_on='id', right_on='game_id')
    
    rankings_df = rankings_df.sort_values("final_adjusted_score", ascending=False).reset_index(drop=True)
    rankings_df['Rank'] = rankings_df.index + 1
    rankings_df['Unadjusted Rank'] = rankings_df['average_score'].rank(method='min', ascending=False).astype(int)

    return rankings_df, critics_with_stats_df
