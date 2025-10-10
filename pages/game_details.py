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
    rankings_df, critics_with_stats_df = calculate_custom_game_rankings(games_df, critics_df, ratings_df)
    global_adjusted_avg = rankings_df['final_adjusted_score'].mean()
    # NEW: Calculate the standard deviation of the adjusted scores
    global_adjusted_std = rankings_df['final_adjusted_score'].std()
    
    # Return all the necessary DataFrames and values
    return games_df, critics_with_stats_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg, global_adjusted_std

# --- Authentication & Page Setup ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Game Details", layout="wide")
st.title("Game Deep Dive")

# --- Database Connection & Data Loading ---
try:
    conn = st.connection("mydb", type="sql")
    # Unpack the new global_adjusted_std value
    games_df, critics_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg, global_adjusted_std = load_data(conn)
except Exception as e:
    st.error(f"Database connection or data loading failed: {e}")
    st.stop()
# --- Game Selection ---
game_map = pd.Series(games_df.id.values, index=games_df.game_name).to_dict()
selected_game_name = st.selectbox("Select a Game to Analyze:", game_map.keys())

# In pages/game_details.py

if selected_game_name:
    # --- Get all pre-calculated data for the selected game ---
    # This 'game_info' Series is now the single source of truth for all stats.
    game_info = rankings_df[rankings_df['game_name'] == selected_game_name].iloc[0]
    game_ratings_df = ratings_df[ratings_df['game_id'] == game_info['id']]

    if game_ratings_df.empty:
        st.warning(f"No ratings have been submitted for **{selected_game_name}** yet.")
        st.stop()

# --- Main Scorecard ---
st.subheader(f"Overall Scorecard for {selected_game_name}")
with st.container(border=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        g_col1, g_col2 = st.columns(2)
        
        # --- Raw Score Gauge and Rank ---
        with g_col1:
            # --- Logic for Delta ---
            raw_delta = game_info['average_score'] - global_avg_score
            raw_threshold = 0.5 * global_std_dev
            if raw_delta > raw_threshold:
                delta_str, delta_color = f"▲ {raw_delta:+.2f}", "#27ae60"  # Green
            elif raw_delta < -raw_threshold:
                delta_str, delta_color = f"▼ {raw_delta:+.2f}", "#c0392b"  # Red
            else:
                delta_str, delta_color = f"~ {raw_delta:+.2f}", "#7f8c8d"  # Gray

            # --- Create Gauge using Annotations for custom layout ---
            raw_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=game_info['average_score'],
                number={'font': {'size': 50}},
                gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                       'steps': [{'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]}
            ))
            raw_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            
            # Add annotations for Title, Delta, and Rank
            raw_gauge.add_annotation(x=0.5, y=1.05, text="Raw Average Score", showarrow=False, font={'size': 16})
            raw_gauge.add_annotation(x=0.5, y=0.35, text=delta_str, showarrow=False, font={'size': 14, 'color': delta_color})
            raw_gauge.add_annotation(x=0.5, y=0.15, text=f"Rank #{game_info['Unadjusted Rank']}", showarrow=False, font={'size': 20})

            st.plotly_chart(raw_gauge, use_container_width=True)

        # --- Adjusted Score Gauge and Rank ---
        with g_col2:
            # --- Logic for Delta ---
            adj_delta = game_info['final_adjusted_score'] - global_adjusted_avg
            adj_threshold = 0.5 * global_adjusted_std
            if adj_delta > adj_threshold:
                adj_delta_str, adj_delta_color = f"▲ {adj_delta:+.2f}", "#27ae60"
            elif adj_delta < -adj_threshold:
                adj_delta_str, adj_delta_color = f"▼ {adj_delta:+.2f}", "#c0392b"
            else:
                adj_delta_str, adj_delta_color = f"~ {adj_delta:+.2f}", "#7f8c8d"

            # --- Create Gauge using Annotations for custom layout ---
            adj_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=game_info['final_adjusted_score'],
                number={'font': {'size': 50}},
                gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                       'steps': [{'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]}
            ))
            adj_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            
            # Add annotations for Title, Delta, and Rank
            adj_gauge.add_annotation(x=0.5, y=1.05, text="Final Adjusted Score", showarrow=False, font={'size': 16})
            adj_gauge.add_annotation(x=0.5, y=0.35, text=adj_delta_str, showarrow=False, font={'size': 14, 'color': adj_delta_color})
            adj_gauge.add_annotation(x=0.5, y=0.15, text=f"Rank #{game_info['Rank']}", showarrow=False, font={'size': 20})

            st.plotly_chart(adj_gauge, use_container_width=True)

    # --- Right-hand metrics ---
    with col2:
        st.metric("Number of Ratings", f"{game_info['number_of_ratings']}")
        play_rate = (game_info['number_of_ratings'] / len(critics_df)) * 100
        st.metric("Play Rate", f"{play_rate:.1f}%")
        st.metric("Controversy (Std Dev)", f"{game_ratings_df['score'].std():.3f}" if game_info['number_of_ratings'] > 1 else "N/A")


    # --- Adjusted Score Breakdown ---
    st.subheader("Ranking Breakdown")
    with st.expander("How the Final Adjusted Score is Calculated", expanded=False):
        st.markdown("The final score is a weighted average that is 'shrunk' towards a baseline, penalizing games for not being widely played.")
        c1, c2, c3, c4 = st.columns(4)
        
        # This section correctly pulls all data from the 'game_info' object
        c1.metric("1. Raw Average", f"{game_info['average_score']:.3f}", help="The simple average of all scores this game received.")
        c2.metric("2. Ratings (n)", f"{game_info['number_of_ratings']}", help="The number of critics who rated this game. This determines the weight of the Raw Average.")
        c3.metric("3. Pessimistic Prior", f"{game_info['pessimistic_prior']:.3f}", help="The baseline this score is pulled towards. It's the average 'pessimistic score' (avg - std dev) of all critics who skipped this game.")
        c4.metric("4. Skippers (C)", f"{game_info['number_of_skippers']}", help="The number of critics who did not rate this game. This determines the weight of the Pessimistic Prior.")
        
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = \frac{(n \times \text{Raw Avg}) + (C \times \text{Pessimistic Prior})}{(n + C)} $$')
        calc_str = f"= (({game_info['number_of_ratings']} × {game_info['average_score']:.3f}) + ({game_info['number_of_skippers']} × {game_info['pessimistic_prior']:.3f})) / ({game_info['number_of_ratings']} + {game_info['number_of_skippers']}) = **{game_info['final_adjusted_score']:.3f}**"
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