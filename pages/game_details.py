# pages/game_details.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Caching, Security, and Data Loading ---
@st.cache_data
def load_data(_conn):
    """
    Loads all necessary data at once and caches it.
    This is much faster and more secure than running separate queries on each interaction.
    """
    games_df = _conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    ratings_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;")

    # Calculate each critic's average score for later comparison
    critic_avg_scores = ratings_df.groupby('critic_id')['score'].mean().reset_index().rename(columns={'score': 'critic_avg_score'})
    critics_df = pd.merge(critics_df, critic_avg_scores, left_on='id', right_on='critic_id', how='left')
    
    # Calculate the global average score across all ratings
    global_avg_score = ratings_df['score'].mean()
    
    return games_df, critics_df, ratings_df, global_avg_score

# --- Authentication & Page Setup ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Game Details", layout="wide")
st.title("🎮 Game Deep Dive")

# --- Database Connection & Data Loading ---
try:
    conn = st.connection("mydb", type="sql")
    games_df, critics_df, ratings_df, global_avg_score = load_data(conn)
except Exception as e:
    st.error(f"Database connection or data loading failed: {e}")
    st.stop()

# --- Game Selection ---
game_map = pd.Series(games_df.id.values, index=games_df.game_name).to_dict()
selected_game_name = st.selectbox("Select a Game to Analyze:", game_map.keys())

if selected_game_name:
    selected_game_id = game_map[selected_game_name]
    
    # --- Filter Data for Selected Game (Fast, in-memory operation) ---
    game_ratings_df = ratings_df[ratings_df['game_id'] == selected_game_id]
    
    if game_ratings_df.empty:
        st.warning(f"No ratings have been submitted for **{selected_game_name}** yet.")
        st.stop()

    # --- Calculate Game-Specific Metrics ---
    avg_score = game_ratings_df['score'].mean()
    controversy = game_ratings_df['score'].std()
    ratings_count = len(game_ratings_df)
    play_rate = (ratings_count / len(critics_df)) * 100

    # --- Main Scorecard ---
    st.subheader(f"Overall Scorecard for {selected_game_name}")
    with st.container(border=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Gauge Chart for Average Score
            gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Score vs. Global Average"},
                gauge = {
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [0, 5], 'color': "#e74c3c"},
                        {'range': [5, 7.5], 'color': "#f1c40f"},
                        {'range': [7.5, 10], 'color': "#2ecc71"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': global_avg_score
                    }}))
            gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge, use_container_width=True)

        with col2:
            st.metric("Number of Ratings", f"{ratings_count}")
            st.metric("Play Rate", f"{play_rate:.1f}%", help="The percentage of all critics who rated this game.")

        with col3:
            st.metric("Controversy (Std Dev)", f"{controversy:.3f}" if controversy else "N/A", help="Standard deviation of scores. Higher means more disagreement.")
            st.metric("Global Average", f"{global_avg_score:.2f}", help="The average score across all ratings for all games.")
    
    # --- Detailed Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "🧑‍⚖️ Critic Ratings", "❓ Who Hasn't Rated?"])

    with tab1:
        st.subheader("How the Scores Broke Down")
        score_counts = game_ratings_df['score'].value_counts().sort_index()
        
        fig = px.bar(
            score_counts, 
            x=score_counts.index, 
            y=score_counts.values,
            labels={'x': 'Score', 'y': 'Number of Critics'},
            text_auto=True
        )
        fig.update_layout(xaxis=dict(tickmode='linear')) # Ensures all integer scores are shown
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight highest and lowest scores
        highest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmax()]
        lowest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmin()]
        highest_critic = critics_df[critics_df['id'] == highest_rating['critic_id']]['critic_name'].iloc[0]
        lowest_critic = critics_df[critics_df['id'] == lowest_rating['critic_id']]['critic_name'].iloc[0]

        tcol1, tcol2 = st.columns(2)
        tcol1.info(f"**Highest Score:** {highest_rating['score']:.1f} by {highest_critic}")
        tcol2.error(f"**Lowest Score:** {lowest_rating['score']:.1f} by {lowest_critic}")
        
    with tab2:
        st.subheader("Individual Scores and Critic Context")
        
        # Merge critic data to get names and their average scores
        detailed_scores_df = pd.merge(game_ratings_df, critics_df, left_on='critic_id', right_on='id')
        detailed_scores_df['delta'] = detailed_scores_df['score'] - detailed_scores_df['critic_avg_score']
        
        st.dataframe(
            detailed_scores_df[['critic_name', 'score', 'critic_avg_score', 'delta']],
            column_config={
                "critic_name": "Critic",
                "score": st.column_config.ProgressColumn("Their Score", min_value=0, max_value=10),
                "critic_avg_score": st.column_config.NumberColumn("Critic's Avg.", help="This critic's average score across all games they've rated.", format="%.2f"),
                "delta": st.column_config.NumberColumn("vs. Their Avg.", help="How this score compares to the critic's personal average.", format="%+.2f")
            },
            hide_index=True, use_container_width=True
        )
        
    with tab3:
        st.subheader("Critics Who Haven't Rated This Game")
        rated_critic_ids = game_ratings_df['critic_id'].unique()
        unrated_critics_df = critics_df[~critics_df['id'].isin(rated_critic_ids)]
        st.dataframe(unrated_critics_df[['critic_name']], hide_index=True, use_container_width=True)