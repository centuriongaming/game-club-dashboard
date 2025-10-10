# pages/critic_details.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating

# --- Page Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")

# --- Authentication and Session ---
check_auth()
session = get_sqla_session()

# --- Cached Data Function ---
@st.cache_data
def get_controversy_data(_session):
    """
    Performs the full, multi-step controversy calculation for ALL critics
    and returns the final scores and the detailed scaffold dataframe.
    This function is cached to prevent re-running on every interaction.
    """
    # 1. Fetch Base Data
    critics_df = pd.read_sql(sa.select(Critic.id, Critic.critic_name), _session.bind)
    all_games_df = pd.read_sql(sa.select(Game.id.label("game_id"), Game.game_name).where(Game.upcoming == False), _session.bind)
    all_ratings_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), _session.bind)

    if all_ratings_df.empty:
        return pd.DataFrame(), pd.DataFrame()

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

    # 7. Calculate Observed Score (Revised Logic to prevent dilution)
    rated_scaffold = scaffold.dropna(subset=['score']).copy()
    avg_score_dev = rated_scaffold.groupby('critic_id')['normalized_score_deviation'].mean().reset_index(name='avg_score_deviation')
    avg_play_dev = scaffold.groupby('critic_id')['play_deviation'].mean().reset_index(name='avg_play_deviation')

    observed_df = pd.merge(critics_df, critic_stats, left_on='id', right_on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_score_dev, on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_play_dev, on='critic_id', how='left')
    observed_df = observed_df.fillna({'n': 0, 'avg_score_deviation': 0, 'avg_play_deviation': 0})
    observed_df['observed_score'] = (0.5 * observed_df['avg_score_deviation']) + (0.5 * observed_df['avg_play_deviation'])
    
    # 8. Apply Bayesian Shrinkage
    prior_score = observed_df['observed_score'].mean()
    C = 15
    observed_df['credibility_weight'] = observed_df['n'] / (observed_df['n'] + C)
    observed_df['final_controversy_score'] = (observed_df['credibility_weight'] * observed_df['observed_score']) + ((1 - observed_df['credibility_weight']) * prior_score)
    observed_df['prior_score'] = prior_score
    observed_df['credibility_constant'] = C

    return observed_df, scaffold

# --- Page Content ---
st.title("Critic Deep Dive")

# --- Run Calculation and Prepare Data ---
controversy_scores_df, scaffold_df = get_controversy_data(session)

if controversy_scores_df.empty:
    st.warning("No rating data available to generate analysis.")
    st.stop()

# --- Critic Selection ---
# Sort critics by name for the dropdown
sorted_critics = controversy_scores_df.sort_values('critic_name')['critic_name'].tolist()
selected_critic_name = st.selectbox("Select a Critic to Analyze:", sorted_critics)

if selected_critic_name:
    # --- Filter pre-calculated data for the selected critic ---
    critic_breakdown = controversy_scores_df.loc[controversy_scores_df['critic_name'] == selected_critic_name].iloc[0]
    details_df = scaffold_df.loc[scaffold_df['critic_name'] == selected_critic_name]
    critic_ratings = details_df.dropna(subset=['score'])
    
    # --- Scorecard Metrics ---
    total_nominations = session.query(sa.func.count(Game.id)).filter(Game.nominated_by == critic_breakdown['id']).scalar()
    participation_rate = critic_breakdown['n'] / details_df['game_id'].nunique() * 100
    avg_score = critic_ratings['score'].mean() if not critic_ratings.empty else 0

    st.subheader(f"Scorecard for {selected_critic_name}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participation Rate", f"{participation_rate:.1f}%", help="Percentage of all available games the critic has rated.")
        col2.metric("Average Score Given", f"{avg_score:.2f}")
        col3.metric("Final Controversy Score", f"{critic_breakdown['final_controversy_score']:.3f}")
        col4.metric("Games Nominated", f"{total_nominations}")

    # --- Controversy Score Breakdown Expander ---
    with st.expander("**Controversy Score Breakdown**", expanded=True):
        st.markdown("The final score is a weighted average of the critic's observed score and the group's average, adjusted for the number of games rated (`n`).")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Observed Score", f"{critic_breakdown['observed_score']:.3f}", help="The critic's raw, un-adjusted controversy score.")
        c2.metric("2. Games Rated (n)", f"{critic_breakdown['n']:.0f}", help="The number of games rated, used to determine credibility.")
        c3.metric("3. Group Average", f"{critic_breakdown['prior_score']:.3f}", help="The average controversy score for the entire group.")
        c4.metric("Credibility Weight", f"{critic_breakdown['credibility_weight']:.1%}", help=f"Weight given to the observed score. Calculated as n / (n + C), where C={critic_breakdown['credibility_constant']}.")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = (\text{Weight} \times \text{Observed}) + (1 - \text{Weight}) \times \text{Group Average} $$')
        calc_str = f"= ({critic_breakdown['credibility_weight']:.2f} × {critic_breakdown['observed_score']:.3f}) + ({1-critic_breakdown['credibility_weight']:.2f} × {critic_breakdown['prior_score']:.3f}) = **{critic_breakdown['final_controversy_score']:.3f}**"
        st.markdown(calc_str)

    # --- Detailed Analysis Tabs ---
    st.subheader("Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Most Contrarian Ratings", "Contrarian Participation", "Full Rating History"])
    
    with tab1:
        st.markdown("These are the games where the critic's score differed most from the group average.")
        st.dataframe(
            critic_ratings.sort_values('normalized_score_deviation', ascending=False).head(10)
            .rename(columns={'score': 'critic_score', 'normalized_score_deviation': 'deviation'})
            [['game_name', 'critic_score', 'avg_game_score', 'deviation']],
            column_config={
                "game_name": "Game",
                "critic_score": "Their Score",
                "avg_game_score": "Group Average",
                "deviation": st.column_config.NumberColumn("Score Deviation (0-1)", format="%.3f")
            },
            hide_index=True, use_container_width=True
        )

    with tab2:
        st.markdown("These are the games where the critic's decision to play or not play went against the grain the most.")
        st.dataframe(
            details_df.sort_values('play_deviation', ascending=False).head(10)
            .rename(columns={'score': 'critic_score'})
            [['game_name', 'critic_score', 'participation_rate']],
            column_config={
                "game_name": "Game",
                "critic_score": "Their Score (if played)",
                "participation_rate": st.column_config.ProgressColumn("Group Participation Rate", format="%.1f%%", min_value=0, max_value=1)
            },
            hide_index=True, use_container_width=True
        )

    with tab3:
        st.markdown("This is the critic's complete rating history for all games.")
        st.dataframe(
            critic_ratings.sort_values('game_name')
            .rename(columns={'score': 'critic_score'})
            [['game_name', 'critic_score', 'avg_game_score']],
            column_config={
                "game_name": "Game",
                "critic_score": "Their Score",
                "avg_game_score": "Group Average"
            },
            hide_index=True, use_container_width=True
        )