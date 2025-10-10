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

    # 7. Calculate Observed Score (Revised Logic)

    # 7a. Calculate the average score deviation ONLY on rated games
    rated_scaffold = scaffold.dropna(subset=['score']).copy()
    avg_score_dev = rated_scaffold.groupby('critic_id')['normalized_score_deviation'].mean().reset_index()
    avg_score_dev = avg_score_dev.rename(columns={'normalized_score_deviation': 'avg_score_deviation'})

    # 7b. Calculate the average play deviation across ALL possible games
    avg_play_dev = scaffold.groupby('critic_id')['play_deviation'].mean().reset_index()
    avg_play_dev = avg_play_dev.rename(columns={'play_deviation': 'avg_play_deviation'})

    # 7c. Merge the two scores together
    observed_df = pd.merge(critics_df, critic_stats, left_on='id', right_on='critic_id')
    observed_df = pd.merge(observed_df, avg_score_dev, on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_play_dev, on='critic_id', how='left')
    observed_df = observed_df.fillna({'avg_score_deviation': 0, 'avg_play_deviation': 0})

    # 7d. Combine them into the final observed score
    observed_df['observed_score'] = (0.5 * observed_df['avg_score_deviation']) + (0.5 * observed_df['avg_play_deviation'])


    # 8. Apply Bayesian Shrinkage
    prior_score = observed_df['observed_score'].mean()
    C = 15
    observed_df['credibility_weight'] = observed_df['n'] / (observed_df['n'] + C)
    observed_df['final_controversy_score'] = (observed_df['credibility_weight'] * observed_df['observed_score']) + ((1 - observed_df['credibility_weight']) * prior_score)
    
    # Add prior_score and C for breakdown display later
    observed_df['prior_score'] = prior_score
    observed_df['credibility_constant'] = C
    
    return observed_df.sort_values('final_controversy_score', ascending=False)

def calculate_custom_game_rankings(games_df, critics_df, ratings_df):
    """
    Calculates game rankings using a personalized Bayesian average.

    The ranking is adjusted based on the number of critics who skipped a game
    and the average "pessimistic score" of those skippers.

    Args:
        games_df (pd.DataFrame): DataFrame with game info (id, game_name).
        critics_df (pd.DataFrame): DataFrame with critic info (id, critic_name).
        ratings_df (pd.DataFrame): DataFrame with all ratings (critic_id, game_id, score).

    Returns:
        pd.DataFrame: A ranked DataFrame with the final adjusted scores.
    """
    # 1. Pre-calculate personal stats for every critic
    critic_stats = ratings_df.groupby('critic_id')['score'].agg(['mean', 'std']).rename(columns={'mean': 'critic_avg', 'std': 'critic_std'})
    critic_stats['critic_std'] = critic_stats['critic_std'].fillna(0) # Std dev is NaN for critics with <=1 rating
    critics_with_stats_df = pd.merge(critics_df, critic_stats, left_on='id', right_on='critic_id', how='left')
    global_avg_fallback = ratings_df['score'].mean() # Fallback for edge cases

    # 2. Define the calculation to be applied to each game
    def calculate_game_rank(game_group):
        # Basic game stats
        n = len(game_group)
        game_avg = game_group['score'].mean()
        
        # Identify skippers and get their stats
        raters_ids = set(game_group['critic_id'])
        all_critic_ids = set(critics_with_stats_df['id'])
        skipper_ids = all_critic_ids - raters_ids
        n_skipped = len(skipper_ids)
        
        # Calculate the pessimistic prior for this specific game
        if n_skipped > 0:
            skipper_stats = critics_with_stats_df[critics_with_stats_df['id'].isin(skipper_ids)]
            # Calculate pessimistic score, handling critics with no ratings (NaN avg/std)
            skipper_stats['pessimistic_score'] = skipper_stats['critic_avg'] - skipper_stats['critic_std']
            pessimistic_prior = skipper_stats['pessimistic_score'].mean()
        else:
            pessimistic_prior = global_avg_fallback
        
        # If the prior is still NaN (e.g., all skippers had no ratings), use global average
        if pd.isna(pessimistic_prior):
            pessimistic_prior = global_avg_fallback

        # 3. Apply the custom formula
        numerator = (n * game_avg) + (n_skipped * pessimistic_prior)
        denominator = n + n_skipped
        adjusted_score = numerator / denominator if denominator > 0 else 0
        
        return pd.Series({
            'number_of_ratings': n,
            'average_score': game_avg,
            'final_adjusted_score': adjusted_score
        })

    # Run the calculation for every game and merge with game names
    rankings_df = ratings_df.groupby('game_id').apply(calculate_game_rank)
    rankings_df = pd.merge(games_df, rankings_df, left_on='id', right_on='game_id')
    
    # 4. Sort and add final rank columns
    rankings_df = rankings_df.sort_values("final_adjusted_score", ascending=False).reset_index(drop=True)
    rankings_df['Rank'] = rankings_df.index + 1
    rankings_df['Unadjusted Rank'] = rankings_df['average_score'].rank(method='min', ascending=False).astype(int)

    return rankings_df
