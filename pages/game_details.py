import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
# Import the ranking function from your utils file
from utils import calculate_custom_game_rankings

# --- Caching, Security, and Data Loading ---
@st.cache_data
def load_data(_conn):
    """
    Loads all necessary data at once and caches it.
    """
    # Load base tables
    games_df = _conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    full_ratings_scaffold_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;")
    ratings_df = full_ratings_scaffold_df.dropna(subset=['score']).copy()
    
    # Call the utility function to get BOTH rankings and critic stats
    rankings_df, critics_with_stats_df = calculate_custom_game_rankings(games_df, critics_df, ratings_df)
    
    # Calculate global stats from the returned data
    global_avg_score = ratings_df['score'].mean()
    global_std_dev = ratings_df['score'].std()
    global_adjusted_avg = rankings_df['final_adjusted_score'].mean()
    
    # Return all the necessary DataFrames
    return games_df, critics_with_stats_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg

# --- Authentication & Page Setup ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Game Details", layout="wide")
st.title("Game Deep Dive")

# --- Database Connection & Data Loading ---
try:
    conn = st.connection("mydb", type="sql")
    games_df, critics_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg = load_data(conn)
except Exception as e:
    st.error(f"Database connection or data loading failed: {e}")
    st.stop()

# --- Game Selection ---
game_map = pd.Series(games_df.id.values, index=games_df.game_name).to_dict()
selected_game_name = st.selectbox("Select a Game to Analyze:", game_map.keys())

if selected_game_name:
    selected_game_id = game_map[selected_game_name]
    game_ratings_df = ratings_df[ratings_df['game_id'] == selected_game_id]
    
    if game_ratings_df.empty:
        st.warning(f"No ratings have been submitted for **{selected_game_name}** yet.")
        st.stop()

    # --- Calculate All Game-Specific Metrics ---
    avg_score = game_ratings_df['score'].mean()
    controversy = game_ratings_df['score'].std()
    ratings_count = len(game_ratings_df)
    
    # Get the game's rank info from the pre-calculated dataframe
    game_rank_info = rankings_df[rankings_df['game_name'] == selected_game_name].iloc[0]
    final_adjusted_score = game_rank_info['final_adjusted_score']
    unadjusted_rank = game_rank_info['Unadjusted Rank']
    adjusted_rank = game_rank_info['Rank']
    
    # --- Main Scorecard ---
    st.subheader(f"Overall Scorecard for {selected_game_name}")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            g_col1, g_col2 = st.columns(2)
            # Gauge 1: Raw Average Score
            with g_col1:
                raw_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta", value = avg_score,
                    delta = {'reference': global_avg_score, 'position': "bottom"},
                    title = {'text': "Raw Average Score"},
                    gauge = { 'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"}, # Blue color
                        'steps': [ {'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]
                    }))
                raw_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(raw_gauge, use_container_width=True)
            # Gauge 2: Final Adjusted Score
            with g_col2:
                adj_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta", value = final_adjusted_score,
                    delta = {'reference': global_adjusted_avg, 'position': "bottom"},
                    title = {'text': "Final Adjusted Score"},
                    gauge = { 'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"}, # Same blue color
                        'steps': [ {'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]
                    }))
                adj_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(adj_gauge, use_container_width=True)

        with col2:
            st.metric("Unadjusted Rank", f"#{unadjusted_rank}")
            st.metric("Adjusted Rank", f"#{adjusted_rank}")
            st.metric("Number of Ratings", f"{ratings_count}")
            st.metric("Controversy (Std Dev)", f"{controversy:.3f}" if controversy else "N/A")


    # --- Adjusted Score Breakdown ---
    st.subheader("Ranking Breakdown")
    with st.expander("How the Final Adjusted Score is Calculated", expanded=False):
        st.markdown("The final score is a weighted average that is 'shrunk' towards a baseline, penalizing games for not being widely played.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Raw Average", f"{avg_score:.3f}", help="The simple average of all scores this game received.")
        c2.metric("2. Ratings (n)", f"{ratings_count}", help="The number of critics who rated this game. This determines the weight of the Raw Average.")
        c3.metric("3. Pessimistic Prior", f"{pessimistic_prior:.3f}", help="The baseline this score is pulled towards. It's the average 'pessimistic score' (avg - std dev) of all critics who skipped this game.")
        c4.metric("4. Skippers (C)", f"{n_skipped}", help="The number of critics who did not rate this game. This determines the weight of the Pessimistic Prior.")
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = \frac{(n \times \text{Raw Avg}) + (C \times \text{Pessimistic Prior})}{(n + C)} $$')
        calc_str = f"= (({ratings_count} × {avg_score:.3f}) + ({n_skipped} × {pessimistic_prior:.3f})) / ({ratings_count} + {n_skipped}) = **{final_adjusted_score:.3f}**"
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
        detailed_scores_df['delta'] = detailed_scores_df['score'] - detailed_scores_df['critic_avg']
        
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
            column_config={
                "critic_name": "Critic",
                "score": st.column_config.ProgressColumn("Their Score", format="%.1f", min_value=0, max_value=10),
                # FIX: Use the correct column name 'critic_avg' here
                "critic_avg": st.column_config.NumberColumn("Critic's Avg.", help="This critic's average score across all games.", format="%.2f"),
                "vs. Their Avg.": "vs. Their Avg."
            },
            hide_index=True, use_container_width=True,
            # FIX: Use the correct column name 'critic_avg' here
            column_order=['critic_name', 'score', 'critic_avg', 'vs. Their Avg.']
        )
        
        st.markdown(f"""<small><b>Legend:</b><br>
            <b>▲ Higher</b>: Score is more than {threshold:.2f} points above the critic's personal average.<br>
            <b>▼ Lower</b>: Score is more than {threshold:.2f} points below their personal average.<br>
            <b>~ About the Same</b>: Score is within {threshold:.2f} points of their personal average.</small>""",
            unsafe_allow_html=True)
        
    with tab3:
        rated_critic_ids = game_ratings_df['critic_id'].unique()
        unrated_critics_df = critics_df[~critics_df['id'].isin(rated_critic_ids)]
        st.dataframe(unrated_critics_df[['critic_name']], hide_index=True, use_container_width=True)