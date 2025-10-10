# pages/game_details.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# --- Caching, Security, and Data Loading ---
@st.cache_data
def load_data(_conn):
    """
    Loads all necessary data at once and caches it.
    This is much faster and more secure than running separate queries on each interaction.
    """
    games_df = _conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    
    # Load the full ratings scaffold from the database
    full_ratings_scaffold_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;")

    # Create the true ratings_df by dropping rows where the score is NULL
    ratings_df = full_ratings_scaffold_df.dropna(subset=['score']).copy()
    
    # Pre-calculate each critic's average and std dev
    critic_stats = ratings_df.groupby('critic_id')['score'].agg(['mean', 'std']).rename(columns={'mean': 'critic_avg', 'std': 'critic_std'}).fillna(0)
    critics_with_stats_df = pd.merge(critics_df, critic_stats, left_on='id', right_on='critic_id', how='left')
    
    # Calculate global stats
    global_avg_score = ratings_df['score'].mean()
    global_std_dev = ratings_df['score'].std()
    
    return games_df, critics_with_stats_df, ratings_df, global_avg_score, global_std_dev

# --- Authentication & Page Setup ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Game Details", layout="wide")
st.title("Game Deep Dive")

# --- Database Connection & Data Loading ---
try:
    conn = st.connection("mydb", type="sql")
    games_df, critics_df, ratings_df, global_avg_score, global_std_dev = load_data(conn)
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
            gauge = go.Figure(go.Indicator(
                mode = "number+gauge+delta", value = avg_score,
                delta = {'reference': global_avg_score, 'position': "bottom"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Score vs. Global Average"},
                gauge = { 'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                    'steps': [ {'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]
                }))
            gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge, use_container_width=True)
        with col2:
            st.metric("Number of Ratings", f"{ratings_count}")
            st.metric("Play Rate", f"{play_rate:.1f}%", help="The percentage of all critics who rated this game.")
        with col3:
            st.metric("Controversy (Std Dev)", f"{controversy:.3f}" if controversy else "N/A", help="Standard deviation of scores. Higher means more disagreement.")
            st.metric("Global Average", f"{global_avg_score:.2f}", help="The average score across all ratings for all games.")

    # --- Adjusted Score Breakdown ---
    st.subheader("Ranking Breakdown")
    with st.expander("How the Final Adjusted Score is Calculated", expanded=False):
        # 1. Basic game stats
        n = ratings_count
        game_avg = avg_score
        
        # 2. Identify skippers and their count
        all_critic_ids = set(critics_df['id'])
        rater_ids = set(game_ratings_df['critic_id'])
        skipper_ids = all_critic_ids - rater_ids
        n_skipped = len(skipper_ids)
        
        # 3. Calculate the Pessimistic Prior
        if n_skipped > 0:
            skipper_stats = critics_df[critics_df['id'].isin(skipper_ids)]
            skipper_stats['pessimistic_score'] = skipper_stats['critic_avg'] - skipper_stats['critic_std']
            pessimistic_prior = skipper_stats['pessimistic_score'].mean()
        else:
            pessimistic_prior = global_avg_score

        # 4. Calculate the Final Score
        final_score = ((n * game_avg) + (n_skipped * pessimistic_prior)) / (n + n_skipped) if (n + n_skipped) > 0 else 0

        st.markdown("The final score is a weighted average that is 'shrunk' towards a baseline, penalizing games for not being widely played.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Raw Average", f"{game_avg:.3f}", help="The simple average of all scores this game received.")
        c2.metric("2. Ratings (n)", f"{n}", help="The number of critics who rated this game. This determines the weight of the Raw Average.")
        c3.metric("3. Pessimistic Prior", f"{pessimistic_prior:.3f}", help="The baseline this score is pulled towards. It's the average 'pessimistic score' (avg - std dev) of all critics who skipped this game.")
        c4.metric("4. Skippers (C)", f"{n_skipped}", help="The number of critics who did not rate this game. This determines the weight of the Pessimistic Prior.")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = \frac{(n \times \text{Raw Avg}) + (C \times \text{Pessimistic Prior})}{(n + C)} $$')
        calc_str = f"= (({n} × {game_avg:.3f}) + ({n_skipped} × {pessimistic_prior:.3f})) / ({n} + {n_skipped}) = **{final_score:.3f}**"
        st.markdown(calc_str)
    
    # --- Detailed Tabs ---
    st.subheader("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Critic Ratings", "Who Skipped?"])

    with tab1:
        scores = game_ratings_df['score']
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(0, 10, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', fill='tozeroy', line_shape='spline', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=scores, y=[0.005] * len(scores), mode='markers', marker=dict(symbol='line-ns-open', color='black', size=10), name='Individual Ratings'))
        fig.update_layout(showlegend=False, xaxis_title="Score", yaxis_title="Density", xaxis=dict(range=[0, 10]))
        st.plotly_chart(fig, use_container_width=True)
        
        highest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmax()]
        lowest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmin()]
        highest_critic = critics_df[critics_df['id'] == highest_rating['critic_id']]['critic_name'].iloc[0]
        lowest_critic = critics_df[critics_df['id'] == lowest_rating['critic_id']]['critic_name'].iloc[0]

        tcol1, tcol2 = st.columns(2)
        tcol1.info(f"**Highest Score:** {highest_rating['score']:.1f} by {highest_critic}")
        tcol2.error(f"**Lowest Score:** {lowest_rating['score']:.1f} by {lowest_critic}")
        
    with tab2:
        detailed_scores_df = pd.merge(game_ratings_df, critics_df, left_on='critic_id', right_on='id')
        detailed_scores_df['delta'] = detailed_scores_df['score'] - detailed_scores_df['critic_avg_score']
        
        threshold = 0.5 * global_std_dev
        def format_delta_with_symbol(delta):
            if delta > threshold: return f"▲ {delta:+.2f}"
            elif delta < -threshold: return f"▼ {delta:+.2f}"
            else: return f"~ {delta:+.2f}"
        def style_delta_column(val_str):
            if val_str.startswith('▲'): return 'color: #27ae60;' # Green
            elif val_str.startswith('▼'): return 'color: #c0392b;' # Red
            else: return 'color: #7f8c8d;' # Gray
        
        detailed_scores_df['vs. Their Avg.'] = detailed_scores_df['delta'].apply(format_delta_with_symbol)
        styled_df = detailed_scores_df.style.map(style_delta_column, subset=['vs. Their Avg.'])
        
        st.dataframe(
            styled_df,
            column_config={ "critic_name": "Critic", "score": st.column_config.ProgressColumn("Their Score", format="%.1f", min_value=0, max_value=10),
                "critic_avg_score": st.column_config.NumberColumn("Critic's Avg.", help="This critic's average score across all games.", format="%.2f"),
                "vs. Their Avg.": "vs. Their Avg."},
            hide_index=True, use_container_width=True,
            column_order=['critic_name', 'score', 'critic_avg_score', 'vs. Their Avg.'])
        
        st.markdown(f"""<small><b>Legend:</b><br>
            <b>▲ Higher</b>: Score is more than {threshold:.2f} points above the critic's personal average.<br>
            <b>▼ Lower</b>: Score is more than {threshold:.2f} points below their personal average.<br>
            <b>~ About the Same</b>: Score is within {threshold:.2f} points of their personal average.</small>""",
            unsafe_allow_html=True)
        
    with tab3:
        rated_critic_ids = game_ratings_df['critic_id'].unique()
        unrated_critics_df = critics_df[~critics_df['id'].isin(rated_critic_ids)]
        st.dataframe(unrated_critics_df[['critic_name']], hide_index=True, use_container_width=True)