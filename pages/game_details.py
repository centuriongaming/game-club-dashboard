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
    games_df = _conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    full_ratings_scaffold_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;")
    ratings_df = full_ratings_scaffold_df.dropna(subset=['score']).copy()
    
    rankings_df, critics_with_stats_df = calculate_custom_game_rankings(games_df, critics_df, ratings_df)
    
    global_avg_score = ratings_df['score'].mean()
    global_std_dev = ratings_df['score'].std()
    global_adjusted_avg = rankings_df['final_adjusted_score'].mean()
    global_adjusted_std = rankings_df['final_adjusted_score'].std()
    
    return games_df, critics_with_stats_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg, global_adjusted_std

# --- Authentication & Page Setup ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Game Details", layout="wide")

# --- FIX 2: Add CSS to center the st.metric values ---
st.markdown("""
<style>
div[data-testid="stMetric"] {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("Game Deep Dive")

# --- Database Connection & Data Loading ---
try:
    conn = st.connection("mydb", type="sql")
    games_df, critics_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg, global_adjusted_std = load_data(conn)
except Exception as e:
    st.error(f"Database connection or data loading failed: {e}")
    st.stop()

# --- Game Selection ---
game_map = pd.Series(games_df.id.values, index=games_df.game_name).to_dict()
selected_game_name = st.selectbox("Select a Game to Analyze:", game_map.keys())

# --- Temporary line to find your column name. You can remove this later. ---
st.write("Available columns for critics:", critics_df.columns) 

if selected_game_name:
    game_info = rankings_df.loc[rankings_df['game_name'] == selected_game_name].iloc[0]
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
            
            with g_col1:
                raw_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=game_info['average_score'],
                    delta={'reference': global_avg_score, 'position': "bottom"},
                    title={'text': "Raw Average Score"},
                    gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                           'steps': [{'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]}
                ))
                raw_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(raw_gauge, use_container_width=True)
                st.metric("Unadjusted Rank", f"#{int(game_info['Unadjusted Rank'])}")

            with g_col2:
                adj_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=game_info['final_adjusted_score'],
                    delta={'reference': global_adjusted_avg, 'position': "bottom"},
                    title={'text': "Final Adjusted Score"},
                    gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                           'steps': [{'range': [0, 5], 'color': "#e74c3c"}, 
                                    {'range': [5, 7.5], 'color': "#f1c40f"}, # Corrected this line
                                    {'range': [7.5, 10], 'color': "#2ecc71"}]}
                ))
                adj_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(adj_gauge, use_container_width=True)
                st.metric("Adjusted Rank", f"#{int(game_info['Rank'])}")

        with col2:
            st.metric("Number of Ratings", f"{game_info['number_of_ratings']}")
            play_rate = (game_info['number_of_ratings'] / len(critics_df)) * 100
            st.metric("Play Rate", f"{play_rate:.1f}%")
            st.metric("Controversy (Std Dev)", f"{game_ratings_df['score'].std():.3f}" if game_info['number_of_ratings'] > 1 else "N/A")

    # --- Adjusted Score Breakdown ---
    # ... (This section is unchanged) ...
    st.subheader("Ranking Breakdown")
    with st.expander("How the Final Adjusted Score is Calculated", expanded=False):
        st.markdown("The final score is a weighted average that is 'shrunk' towards a baseline, penalizing games for not being widely played.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Raw Average", f"{game_info['average_score']:.3f}")
        c2.metric("2. Ratings (n)", f"{game_info['number_of_ratings']}")
        c3.metric("3. Pessimistic Prior", f"{game_info['pessimistic_prior']:.3f}")
        c4.metric("4. Skippers (C)", f"{game_info['number_of_skippers']}")
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = \frac{(n \times \text{Raw Avg}) + (C \times \text{Pessimistic Prior})}{(n + C)} $$')
        calc_str = f"= (({game_info['number_of_ratings']} × {game_info['average_score']:.3f}) + ({game_info['number_of_skippers']} × {game_info['pessimistic_prior']:.3f})) / ({game_info['number_of_ratings']} + {game_info['number_of_skippers']}) = **{game_info['final_adjusted_score']:.3f}**"
        st.markdown(calc_str)
    
    # --- Detailed Tabs ---
    st.subheader("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Critic Ratings", "Who Skipped?"])
    
    with tab1:
        # ... (This tab is unchanged) ...
        if len(game_ratings_df['score']) > 1:
            scores = game_ratings_df['score']
            kde = stats.gaussian_kde(scores)
            x_range = np.linspace(0, 10, 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', fill='tozeroy', line_shape='spline', line=dict(color='#3498db')))
            fig.add_trace(go.Scatter(x=scores, y=[0.005] * len(scores), mode='markers', marker=dict(symbol='line-ns-open', color='black', size=10), name='Individual Ratings'))
            fig.update_layout(showlegend=False, xaxis_title="Score", yaxis_title="Density", xaxis=dict(range=[0, 10]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A score distribution plot cannot be generated with fewer than two ratings.")

        highest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmax()]
        lowest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmin()]
        highest_critic = critics_df.loc[critics_df['id'] == highest_rating['critic_id'], 'critic_name'].iloc[0]
        lowest_critic = critics_df.loc[critics_df['id'] == lowest_rating['critic_id'], 'critic_name'].iloc[0]

        tcol1, tcol2 = st.columns(2)
        tcol1.info(f"**Highest Score:** {highest_rating['score']:.1f} by {highest_critic}")
        tcol2.error(f"**Lowest Score:** {lowest_rating['score']:.1f} by {lowest_critic}")
        
    with tab2:
        detailed_scores_df = pd.merge(game_ratings_df, critics_df, left_on='critic_id', right_on='id')
        
        # --- FIX 1: Replace 'critic_avg_score' with the correct column name you found! ---
        placeholder_col_name = 'critic_avg_score' # <-- CHANGE THIS
        # ----------------------------------------------------------------------------------

        detailed_scores_df['delta'] = detailed_scores_df['score'] - detailed_scores_df[placeholder_col_name]
        
        threshold = 0.5 * global_std_dev
        def format_delta_with_symbol(delta):
            if delta > threshold: return f"▲ {delta:+.2f}"
            elif delta < -threshold: return f"▼ {delta:+.2f}"
            else: return f"~ {delta:+.2f}"
        def style_delta_column(val_str):
            if val_str.startswith('▲'): return 'color: #27ae60;'
            elif val_str.startswith('▼'): return 'color: #c0392b;'
            else: return 'color: #7f8c8d;'
        
        detailed_scores_df['vs. Their Avg.'] = detailed_scores_df['delta'].apply(format_delta_with_symbol)
        styled_df = detailed_scores_df.style.map(style_delta_column, subset=['vs. Their Avg.'])
        
        st.dataframe(
            styled_df,
            column_config={
                "critic_name": "Critic",
                "score": st.column_config.ProgressColumn("Their Score", format="%.1f", min_value=0, max_value=10),
                placeholder_col_name: st.column_config.NumberColumn("Critic's Avg.", help="This critic's average score across all games.", format="%.2f"),
                "vs. Their Avg.": "vs. Their Avg."
            },
            hide_index=True, use_container_width=True,
            column_order=['critic_name', 'score', placeholder_col_name, 'vs. Their Avg.']
        )
        
        st.markdown(f"""<small><b>Legend:</b><br>
            <b>▲ Higher</b>: Score is more than {threshold:.2f} points above the critic's personal average.<br>
            <b>▼ Lower</b>: Score is more than {threshold:.2f} points below their personal average.<br>
            <b>~ About the Same</b>: Score is within {threshold:.2f} points of their personal average.</small>""",
            unsafe_allow_html=True)
        
    with tab3:
        rated_critic_ids = game_ratings_df['critic_id'].unique()
        unrated_critics_df = critics_df[~critics_df['id'].isin(rated_critic_ids)]
        # Display critic name and their pessimistic score, if available
        cols_to_display = ['critic_name']
        if 'critic_avg_pessimistic' in unrated_critics_df.columns:
            cols_to_display.append('critic_avg_pessimistic')

        st.dataframe(unrated_critics_df[cols_to_display], hide_index=True, use_container_width=True)import streamlit as st
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
    games_df = _conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    full_ratings_scaffold_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;")
    ratings_df = full_ratings_scaffold_df.dropna(subset=['score']).copy()
    
    rankings_df, critics_with_stats_df = calculate_custom_game_rankings(games_df, critics_df, ratings_df)
    
    global_avg_score = ratings_df['score'].mean()
    global_std_dev = ratings_df['score'].std()
    global_adjusted_avg = rankings_df['final_adjusted_score'].mean()
    global_adjusted_std = rankings_df['final_adjusted_score'].std()
    
    return games_df, critics_with_stats_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg, global_adjusted_std

# --- Authentication & Page Setup ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Game Details", layout="wide")

# --- FIX 2: Add CSS to center the st.metric values ---
st.markdown("""
<style>
div[data-testid="stMetric"] {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("Game Deep Dive")

# --- Database Connection & Data Loading ---
try:
    conn = st.connection("mydb", type="sql")
    games_df, critics_df, ratings_df, global_avg_score, global_std_dev, rankings_df, global_adjusted_avg, global_adjusted_std = load_data(conn)
except Exception as e:
    st.error(f"Database connection or data loading failed: {e}")
    st.stop()

# --- Game Selection ---
game_map = pd.Series(games_df.id.values, index=games_df.game_name).to_dict()
selected_game_name = st.selectbox("Select a Game to Analyze:", game_map.keys())

# --- Temporary line to find your column name. You can remove this later. ---
st.write("Available columns for critics:", critics_df.columns) 

if selected_game_name:
    game_info = rankings_df.loc[rankings_df['game_name'] == selected_game_name].iloc[0]
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
            
            with g_col1:
                raw_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=game_info['average_score'],
                    delta={'reference': global_avg_score, 'position': "bottom"},
                    title={'text': "Raw Average Score"},
                    gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                           'steps': [{'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]}
                ))
                raw_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(raw_gauge, use_container_width=True)
                st.metric("Unadjusted Rank", f"#{int(game_info['Unadjusted Rank'])}")

            with g_col2:
                adj_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=game_info['final_adjusted_score'],
                    delta={'reference': global_adjusted_avg, 'position': "bottom"},
                    title={'text': "Final Adjusted Score"},
                    gauge={'axis': {'range': [0, 10]}, 'bar': {'color': "#3498db"},
                           'steps': [{'range': [0, 5], 'color': "#e74c3c"}, {'range': [5, 7.5], 'color': "#f1c40f"}, {'range': [7.5, 10], 'color': "#2ecc71"}]}
                ))
                adj_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(adj_gauge, use_container_width=True)
                st.metric("Adjusted Rank", f"#{int(game_info['Rank'])}")

        with col2:
            st.metric("Number of Ratings", f"{game_info['number_of_ratings']}")
            play_rate = (game_info['number_of_ratings'] / len(critics_df)) * 100
            st.metric("Play Rate", f"{play_rate:.1f}%")
            st.metric("Controversy (Std Dev)", f"{game_ratings_df['score'].std():.3f}" if game_info['number_of_ratings'] > 1 else "N/A")

    # --- Adjusted Score Breakdown ---
    # ... (This section is unchanged) ...
    st.subheader("Ranking Breakdown")
    with st.expander("How the Final Adjusted Score is Calculated", expanded=False):
        st.markdown("The final score is a weighted average that is 'shrunk' towards a baseline, penalizing games for not being widely played.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1. Raw Average", f"{game_info['average_score']:.3f}")
        c2.metric("2. Ratings (n)", f"{game_info['number_of_ratings']}")
        c3.metric("3. Pessimistic Prior", f"{game_info['pessimistic_prior']:.3f}")
        c4.metric("4. Skippers (C)", f"{game_info['number_of_skippers']}")
        st.markdown("---")
        st.markdown("##### Final Calculation")
        st.markdown(r'$$ \text{Final Score} = \frac{(n \times \text{Raw Avg}) + (C \times \text{Pessimistic Prior})}{(n + C)} $$')
        calc_str = f"= (({game_info['number_of_ratings']} × {game_info['average_score']:.3f}) + ({game_info['number_of_skippers']} × {game_info['pessimistic_prior']:.3f})) / ({game_info['number_of_ratings']} + {game_info['number_of_skippers']}) = **{game_info['final_adjusted_score']:.3f}**"
        st.markdown(calc_str)
    
    # --- Detailed Tabs ---
    st.subheader("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Critic Ratings", "Who Skipped?"])
    
    with tab1:
        # ... (This tab is unchanged) ...
        if len(game_ratings_df['score']) > 1:
            scores = game_ratings_df['score']
            kde = stats.gaussian_kde(scores)
            x_range = np.linspace(0, 10, 100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', fill='tozeroy', line_shape='spline', line=dict(color='#3498db')))
            fig.add_trace(go.Scatter(x=scores, y=[0.005] * len(scores), mode='markers', marker=dict(symbol='line-ns-open', color='black', size=10), name='Individual Ratings'))
            fig.update_layout(showlegend=False, xaxis_title="Score", yaxis_title="Density", xaxis=dict(range=[0, 10]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A score distribution plot cannot be generated with fewer than two ratings.")

        highest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmax()]
        lowest_rating = game_ratings_df.loc[game_ratings_df['score'].idxmin()]
        highest_critic = critics_df.loc[critics_df['id'] == highest_rating['critic_id'], 'critic_name'].iloc[0]
        lowest_critic = critics_df.loc[critics_df['id'] == lowest_rating['critic_id'], 'critic_name'].iloc[0]

        tcol1, tcol2 = st.columns(2)
        tcol1.info(f"**Highest Score:** {highest_rating['score']:.1f} by {highest_critic}")
        tcol2.error(f"**Lowest Score:** {lowest_rating['score']:.1f} by {lowest_critic}")
        
    with tab2:
        detailed_scores_df = pd.merge(game_ratings_df, critics_df, left_on='critic_id', right_on='id')
        
        # --- FIX 1: Replace 'critic_avg_score' with the correct column name you found! ---
        placeholder_col_name = 'critic_avg_score' # <-- CHANGE THIS
        # ----------------------------------------------------------------------------------

        detailed_scores_df['delta'] = detailed_scores_df['score'] - detailed_scores_df[placeholder_col_name]
        
        threshold = 0.5 * global_std_dev
        def format_delta_with_symbol(delta):
            if delta > threshold: return f"▲ {delta:+.2f}"
            elif delta < -threshold: return f"▼ {delta:+.2f}"
            else: return f"~ {delta:+.2f}"
        def style_delta_column(val_str):
            if val_str.startswith('▲'): return 'color: #27ae60;'
            elif val_str.startswith('▼'): return 'color: #c0392b;'
            else: return 'color: #7f8c8d;'
        
        detailed_scores_df['vs. Their Avg.'] = detailed_scores_df['delta'].apply(format_delta_with_symbol)
        styled_df = detailed_scores_df.style.map(style_delta_column, subset=['vs. Their Avg.'])
        
        st.dataframe(
            styled_df,
            column_config={
                "critic_name": "Critic",
                "score": st.column_config.ProgressColumn("Their Score", format="%.1f", min_value=0, max_value=10),
                placeholder_col_name: st.column_config.NumberColumn("Critic's Avg.", help="This critic's average score across all games.", format="%.2f"),
                "vs. Their Avg.": "vs. Their Avg."
            },
            hide_index=True, use_container_width=True,
            column_order=['critic_name', 'score', placeholder_col_name, 'vs. Their Avg.']
        )
        
        st.markdown(f"""<small><b>Legend:</b><br>
            <b>▲ Higher</b>: Score is more than {threshold:.2f} points above the critic's personal average.<br>
            <b>▼ Lower</b>: Score is more than {threshold:.2f} points below their personal average.<br>
            <b>~ About the Same</b>: Score is within {threshold:.2f} points of their personal average.</small>""",
            unsafe_allow_html=True)
        
    with tab3:
        rated_critic_ids = game_ratings_df['critic_id'].unique()
        unrated_critics_df = critics_df[~critics_df['id'].isin(rated_critic_ids)]
        # Display critic name and their pessimistic score, if available
        cols_to_display = ['critic_name']
        if 'critic_avg_pessimistic' in unrated_critics_df.columns:
            cols_to_display.append('critic_avg_pessimistic')

        st.dataframe(unrated_critics_df[cols_to_display], hide_index=True, use_container_width=True)