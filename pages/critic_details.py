# pages/critic_details.py
import streamlit as st
import pandas as pd
from utils import check_auth, get_db_connection, load_queries

# --- Initial Setup ---
check_auth()
conn = get_db_connection()
queries = load_queries()

# --- Page Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")
st.title("Critic Deep Dive")

# --- Critic Selection ---
critics_list_df = conn.query(queries['get_critics_list'])
critic_names = critics_list_df['critic_name'].tolist()
selected_critic_name = st.selectbox("Select a Critic to Analyze:", critic_names)

if selected_critic_name:
    # --- Data Fetching ---
    critic_id_df = conn.query(f"SELECT id FROM critics WHERE critic_name = '{selected_critic_name}';")
    if critic_id_df.empty:
        st.error("Could not find critic.")
        st.stop()
    selected_critic_id = critic_id_df['id'][0]

    params = {'critic_id': selected_critic_id}
    details_df = conn.query(queries['get_critic_details'], params=params)
    nominations_df = conn.query(queries['get_critic_nominations_count'], params=params)
    
    # Fetch the full controversy breakdown data
    controversy_breakdown_df = conn.query(queries['get_critic_controversy_breakdown'])
    critic_breakdown = controversy_breakdown_df[controversy_breakdown_df['critic_name'] == selected_critic_name].iloc[0]

    # --- Calculations for Scorecard ---
    critic_ratings = details_df.dropna(subset=['critic_score'])
    participation_rate = (len(critic_ratings) / len(details_df)) * 100 if len(details_df) > 0 else 0
    avg_score = critic_ratings['critic_score'].mean() if not critic_ratings.empty else 0
    total_nominations = nominations_df['count'][0]

    # --- Critic Scorecard ---
    st.subheader(f"Scorecard for {selected_critic_name}")
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Participation Rate", f"{participation_rate:.1f}%")
        col2.metric("Average Score Given", f"{avg_score:.2f}")
        col3.metric("Final Controversy Score", f"{critic_breakdown['controversy_score']:.3f}")
        col4.metric("Games Nominated", f"{total_nominations}")

    # --- Controversy Score Breakdown Expander ---
    with st.expander("**Controversy Score Breakdown**"):
        # Extract components from the breakdown dataframe
        observed_score = critic_breakdown['observed_score']
        n = critic_breakdown['n']
        prior_score = critic_breakdown['prior_score']
        C = critic_breakdown['credibility_constant']

        # Calculate weight
        credibility_weight = n / (n + C)

        st.markdown("The final score is a weighted average of the critic's observed score and the group's average, adjusted for the number of games rated (`n`).")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("1. Observed Score", f"{observed_score:.3f}", help="The critic's raw, un-adjusted controversy score.")
        col2.metric("2. Games Rated (n)", f"{n}", help="The number of games rated by the critic, used to determine credibility.")
        col3.metric("3. Group Average", f"{prior_score:.3f}", help="The average controversy score for the entire group.")
        col4.metric("Credibility Weight", f"{credibility_weight:.1%}", help=f"The weight given to the observed score. Calculated as n / (n + C), where C is {C}.")

        st.markdown("---")
        st.markdown("##### Final Calculation")
        formula = r'''
        \text{Final Score} = (\text{Weight} \times \text{Observed}) + (1 - \text{Weight}) \times \text{Group Average}
        '''
        st.latex(formula)

        calculation_str = f"= ({credibility_weight:.2f} \times {observed_score:.3f}) + ({1-credibility_weight:.2f}) \times {prior_score:.3f} = {critic_breakdown['controversy_score']:.3f}"
        st.markdown(f"**{calculation_str}**")


    # --- Detailed Analysis Tabs ---
    st.subheader("Detailed Analysis")
    
    critic_ratings['deviation'] = (critic_ratings['critic_score'] - critic_ratings['avg_game_score']).abs()
    most_contrarian_ratings = critic_ratings.sort_values('deviation', ascending=False).head(10)

    details_df['play_decision_diff'] = (details_df['critic_score'].notna().astype(int) - details_df['participation_rate']).abs()
    most_contrarian_plays = details_df.sort_values('play_decision_diff', ascending=False).head(10)

    tab1, tab2, tab3 = st.tabs(["Most Contrarian Ratings", "Contrarian Participation", "Full Rating History"])

    with tab1:
        # (Content for tab1 remains the same)
        st.markdown("These are the games where the critic's score differed most from the group average.")
        st.dataframe(
            most_contrarian_ratings[['game_name', 'critic_score', 'avg_game_score', 'deviation']],
            column_config={
                "game_name": "Game", "critic_score": "Their Score",
                "avg_game_score": "Group Average", "deviation": st.column_config.BarChartColumn("Deviation")
            },
            hide_index=True, use_container_width=True
        )

    with tab2:
        # (Content for tab2 remains the same)
        st.markdown("These are the games where the critic's decision to play or not play went against the grain the most.")
        st.dataframe(
            most_contrarian_plays[['game_name', 'critic_score', 'participation_rate']],
            column_config={
                "game_name": "Game", "critic_score": "Their Score (if played)",
                "participation_rate": st.column_config.ProgressColumn("Group Participation Rate", format="%.0f%%", min_value=0, max_value=100)
            },
            hide_index=True, use_container_width=True
        )

    with tab3:
        # (Content for tab3 remains the same)
        st.markdown("This is the critic's complete rating history for all games.")
        st.dataframe(
            critic_ratings[['game_name', 'critic_score', 'avg_game_score']],
            column_config={
                "game_name": "Game", "critic_score": "Their Score",
                "avg_game_score": "Group Average"
            },
            hide_index=True, use_container_width=True
        )