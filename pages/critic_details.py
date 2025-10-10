# pages/critic_details.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
from utils import check_auth, get_sqla_session, calculate_controversy_scores
from database_models import Critic, Game, Rating

# --- Page & Data Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")

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

        threshold = critic_breakdown['credibility_threshold']
        weight = games_rated_count / (games_rated_count + threshold) if (games_rated_count + threshold) > 0 else 0
        
        # Use the final score directly from the pre-calculated data.
        final_score = critic_breakdown['final_controversy_score']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "1. Observed Score", 
            f"{critic_breakdown['observed_score']:.3f}", 
            help="A 50/50 average of Score Deviation and Play Deviation. For critics with fewer than 10 reviews, Play Deviation is 0, which intentionally pulls this score closer to zero."
        )
        c2.metric("2. Games Rated (n)", f"{games_rated_count:.0f}", help="The number of games rated, used to calculate credibility.")
        c3.metric("3. Group Average", f"{critic_breakdown['prior_score']:.3f}", help="The average controversy score of all critics, used as a baseline.")
        c4.metric("Credibility Weight", f"{weight:.1%}", help=f"The weight given to the critic's Observed Score. Calculated as n / (n + Threshold), where the threshold is dynamically set to {threshold} (half the total games).")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = (\text{Weight} \times \text{Observed}) + ((1 - \text{Weight}) \times \text{Group Average}) $$')

        # This table uses the final score to derive its parts, ensuring the sum always matches.
        part1 = weight * critic_breakdown['observed_score']
        part2 = final_score - part1
        
        calculation_table = f"""
        | Component | Weight | Value | Result |
        | :--- | :--- | :--- | :--- |
        | **Observed Score** | `{weight:.1%}` | × `{critic_breakdown['observed_score']:.3f}` | `= {part1:.3f}` |
        | **Group Average** | `{1-weight:.1%}`| × `{critic_breakdown['prior_score']:.3f}` | `= {part2:.3f}` |
        | **Final Score** | | **Sum →** | **`= {final_score:.3f}`** |
        """
        st.markdown(calculation_table)
1
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

    # Call the unified function from utils.py
    controversy_df, scaffold_df = calculate_controversy_scores(session)

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
        critic_id = int(critic_breakdown['critic_id'])
        
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