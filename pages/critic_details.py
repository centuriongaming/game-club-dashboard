# pages/critic_details.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating

# --- Page & Data Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")

@st.cache_data
def load_page_data(_session):
    """
    Performs the full controversy calculation for all critics and returns the final
    scores and the detailed scaffold dataframe.
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

    # 7. Calculate Observed Score
    rated_scaffold = scaffold.dropna(subset=['score']).copy()
    avg_score_dev = rated_scaffold.groupby('critic_id')['normalized_score_deviation'].mean().reset_index(name='avg_score_deviation')
    avg_play_dev = scaffold.groupby('critic_id')['play_deviation'].mean().reset_index(name='avg_play_deviation')

    observed_df = pd.merge(critics_df, critic_stats, left_on='id', right_on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_score_dev, on='critic_id', how='left')
    observed_df = pd.merge(observed_df, avg_play_dev, on='critic_id', how='left')
    observed_df = observed_df.fillna({'n': 0, 'avg_score_deviation': 0, 'avg_play_deviation': 0})
    observed_df['observed_score'] = (0.5 * observed_df['avg_score_deviation']) + (0.5 * observed_df['avg_play_deviation'])
    
    # --- THIS SECTION IS UPDATED ---
    # 8. Apply Bayesian Shrinkage
    prior_score = observed_df['observed_score'].mean()
    
    # Dynamically calculate the credibility threshold
    total_games = len(all_games_df)
    credibility_threshold = max(1, total_games // 2)

    observed_df['credibility_weight'] = observed_df['n'] / (observed_df['n'] + credibility_threshold)
    observed_df['final_controversy_score'] = (observed_df['credibility_weight'] * observed_df['observed_score']) + ((1 - observed_df['credibility_weight']) * prior_score)
    observed_df['prior_score'] = prior_score
    observed_df['credibility_threshold'] = credibility_threshold # Store the dynamic value

    return observed_df, scaffold

# --- UI Component Functions ---
def display_scorecard(critic_name, metrics):
    """Renders the main scorecard for the selected critic."""
    st.subheader(f"Scorecard for {critic_name}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participation Rate", f"{metrics['participation_rate']:.1f}%", help="Percentage of all available games the critic has rated.")
        col2.metric("Average Score Given", f"{metrics['avg_score']:.2f}")
        col3.metric("Final Controversy Score", f"{metrics['controversy_score']:.3f}")
        col4.metric("Games Nominated", f"{metrics['nominations']}")

def display_controversy_breakdown(critic_breakdown, games_rated_count):
    """Renders the expander that explains the controversy score calculation."""
    with st.expander("**Controversy Score Breakdown**", expanded=True):
        st.markdown("The final score is a weighted average of the critic's observed score and the group's average, adjusted for the number of games rated (`n`).")

        # Use the new 'credibility_threshold' column
        threshold = critic_breakdown['credibility_threshold']
        weight = games_rated_count / (games_rated_count + threshold) if (games_rated_count + threshold) > 0 else 0
        final_score = (weight * critic_breakdown['observed_score']) + ((1 - weight) * critic_breakdown['prior_score'])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Observed Score", f"{critic_breakdown['observed_score']:.3f}", help="A raw measure of this critic's tendency to deviate from the group, based on both their rating scores and participation habits.")
        c2.metric("2. Games Rated (n)", f"{games_rated_count:.0f}", help="The number of games rated, used to calculate credibility.")
        c3.metric("3. Group Average", f"{critic_breakdown['prior_score']:.3f}", help="The average controversy score of all critics, used as a baseline.")
        # Update the help text to explain the dynamic threshold
        c4.metric("Credibility Weight", f"{weight:.1%}", help=f"The weight given to the critic's Observed Score. Calculated as n / (n + Threshold), where the threshold is dynamically set to {threshold} (half the total games).")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = (\text{Weight} \times \text{Observed}) + (1 - \text{Weight}) \times \text{Group Average} $$')
        calc_str = f"= ({weight:.2f} × {critic_breakdown['observed_score']:.3f}) + ({1-weight:.2f} × {critic_breakdown['prior_score']:.3f}) = **{final_score:.3f}**"
        st.markdown(calc_str)
    
def display_analysis_tabs(critic_ratings, details_df):
    """Renders the three tabs with detailed breakdown tables."""
    st.subheader("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Most Contrarian Ratings", "Contrarian Participation", "Full Rating History"])
    
    with tab1:
        st.markdown("These are the games where the critic's score differed most from the group average.")
        df1 = critic_ratings.sort_values('normalized_score_deviation', ascending=False).head(10)
        st.dataframe(
            df1.rename(columns={'score': 'critic_score', 'normalized_score_deviation': 'deviation'}),
            column_order=['game_name', 'critic_score', 'avg_game_score', 'deviation'],
            column_config={
                "game_name": "Game", "critic_score": "Their Score", "avg_game_score": "Group Average",
                "deviation": st.column_config.NumberColumn("Score Deviation (0-1)", format="%.3f")
            }, hide_index=True, use_container_width=True
        )

    with tab2:
        st.markdown("These are the games where the critic's decision to play or not play went against the grain the most.")
        df2 = details_df.sort_values('play_deviation', ascending=False).head(10)
        st.dataframe(
            df2.rename(columns={'score': 'critic_score'}),
            column_order=['game_name', 'critic_score', 'participation_rate'],
            column_config={
                "game_name": "Game", "critic_score": "Their Score (if played)",
                "participation_rate": st.column_config.ProgressColumn("Group Participation Rate", format="%.1f%%", min_value=0, max_value=1)
            }, hide_index=True, use_container_width=True
        )

    with tab3:
        st.markdown("This is the critic's complete rating history for all games.")
        df3 = critic_ratings.sort_values('game_name')
        st.dataframe(
            df3.rename(columns={'score': 'critic_score'}),
            column_order=['game_name', 'critic_score', 'avg_game_score'],
            column_config={"game_name": "Game", "critic_score": "Their Score", "avg_game_score": "Group Average"},
            hide_index=True, use_container_width=True
        )

# --- Main Page ---
def main():
    """Renders the Critic Details page."""
    check_auth()
    session = get_sqla_session()
    
    st.title("Critic Details")

    controversy_df, scaffold_df = load_page_data(session)

    if controversy_df.empty:
        st.warning("No rating data available to generate analysis.")
        st.stop()

    sorted_critics = controversy_df.sort_values('critic_name')['critic_name'].tolist()
    selected_critic_name = st.selectbox("Select a Critic to Analyze:", sorted_critics)

    if selected_critic_name:
        # Filter data for the selected critic
        critic_breakdown = controversy_df.loc[controversy_df['critic_name'] == selected_critic_name].iloc[0]
        details_df = scaffold_df.loc[scaffold_df['critic_name'] == selected_critic_name]
        critic_ratings = details_df.dropna(subset=['score'])
        
        # Calculate scorecard metrics
        games_rated_count = len(critic_ratings)
        total_games_count = details_df['game_id'].nunique()
        critic_id = int(critic_breakdown['id'])
        
        metrics = {
            "participation_rate": (games_rated_count / total_games_count) * 100 if total_games_count > 0 else 0,
            "avg_score": critic_ratings['score'].mean() if not critic_ratings.empty else 0,
            "controversy_score": critic_breakdown['final_controversy_score'],
            "nominations": session.query(sa.func.count(Game.id)).filter(Game.nominated_by == critic_id).scalar()
        }

        # Render UI components
        display_scorecard(selected_critic_name, metrics)
        display_controversy_breakdown(critic_breakdown, games_rated_count)
        display_analysis_tabs(critic_ratings, details_df)

if __name__ == "__main__":
    main()