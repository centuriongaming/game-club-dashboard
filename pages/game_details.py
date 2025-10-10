import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from utils import calculate_custom_game_rankings

# --- Constants ---
GAUGE_COLORS = {
    'steps': [
        {'range': [0, 5], 'color': "#e74c3c"},
        {'range': [5, 7.5], 'color': "#f1c40f"},
        {'range': [7.5, 10], 'color': "#2ecc71"}
    ],
    'bar': {'color': "#3498db"}
}

# --- Data Loading ---
@st.cache_data
def load_data(_conn):
    """
    Loads all necessary data from the database, performs ranking calculations,
    and caches the results.
    """
    games_df = _conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    ratings_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;").dropna(subset=['score']).copy()

    rankings_df, critics_with_stats_df = calculate_custom_game_rankings(games_df, critics_df, ratings_df.copy())
    
    global_stats = {
        'avg_score': ratings_df['score'].mean(),
        'std_dev': ratings_df['score'].std(),
        'adjusted_avg': rankings_df['final_adjusted_score'].mean(),
    }
    
    return games_df, critics_with_stats_df, ratings_df, rankings_df, global_stats

# --- UI Component Functions ---

def create_gauge_figure(value, reference, title):
    """Creates a Plotly gauge figure."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={'reference': reference, 'position': "bottom"},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': GAUGE_COLORS['bar'],
            'steps': GAUGE_COLORS['steps']
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
    return fig

def display_scorecard(game_info, game_ratings_df, num_total_critics, global_stats):
    """Displays the main scorecard section with gauges and key metrics."""
    st.subheader(f"Overall Scorecard for {game_info['game_name']}")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.plotly_chart(
                    create_gauge_figure(game_info['average_score'], global_stats['avg_score'], "Raw Average Score"),
                    use_container_width=True
                )
                st.metric("Unadjusted Rank", f"#{int(game_info['Unadjusted Rank'])}")
            with g_col2:
                st.plotly_chart(
                    create_gauge_figure(game_info['final_adjusted_score'], global_stats['adjusted_avg'], "Final Adjusted Score"),
                    use_container_width=True
                )
                st.metric("Adjusted Rank", f"#{int(game_info['Rank'])}")
        with col2:
            st.metric("Number of Ratings", f"{game_info['number_of_ratings']}")
            play_rate = (game_info['number_of_ratings'] / num_total_critics) * 100
            st.metric("Play Rate", f"{play_rate:.1f}%")
            std_dev = game_ratings_df['score'].std() if game_info['number_of_ratings'] > 1 else "N/A"
            st.metric("Controversy (Std Dev)", f"{std_dev:.3f}" if isinstance(std_dev, float) else std_dev)

def display_ranking_breakdown(game_info):
    """Displays the expander explaining the Bayesian shrinkage calculation."""
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
        calc_str = (
            f"= (({game_info['number_of_ratings']} × {game_info['average_score']:.3f}) + "
            f"({game_info['number_of_skippers']} × {game_info['pessimistic_prior']:.3f})) / "
            f"({game_info['number_of_ratings']} + {game_info['number_of_skippers']}) = "
            f"**{game_info['final_adjusted_score']:.3f}**"
        )
        st.markdown(calc_str)

def display_score_distribution(game_ratings_with_critics):
    """Displays the score distribution plot and highest/lowest score info."""
    if len(game_ratings_with_critics) > 1:
        scores = game_ratings_with_critics['score']
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(0, 10, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', fill='tozeroy', line_shape='spline', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=scores, y=[0.005] * len(scores), mode='markers', marker=dict(symbol='line-ns-open', color='black', size=10), name='Individual Ratings'))
        fig.update_layout(showlegend=False, xaxis_title="Score", yaxis_title="Density", xaxis=dict(range=[0, 10]))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("A score distribution plot cannot be generated with fewer than two ratings.")

    highest = game_ratings_with_critics.loc[game_ratings_with_critics['score'].idxmax()]
    lowest = game_ratings_with_critics.loc[game_ratings_with_critics['score'].idxmin()]
    
    tcol1, tcol2 = st.columns(2)
    tcol1.info(f"**Highest Score:** {highest['score']:.1f} by {highest['critic_name']}")
    tcol2.error(f"**Lowest Score:** {lowest['score']:.1f} by {lowest['critic_name']}")

def display_critic_ratings(game_ratings_with_critics, global_std_dev):
    """Displays the detailed table of critic ratings vs. their average."""
    df = game_ratings_with_critics.copy()
    
    # **NOTE**: Assumes the critic stats df has this column. Change if the name is different.
    critic_avg_col = 'critic_avg' 
    df['delta'] = df['score'] - df[critic_avg_col]
    
    threshold = 0.5 * global_std_dev

    def format_delta(delta):
        if delta > threshold: return f"▲ {delta:+.2f}"
        if delta < -threshold: return f"▼ {delta:+.2f}"
        return f"~ {delta:+.2f}"

    def style_delta(val_str):
        if val_str.startswith('▲'): return 'color: #27ae60;' # Green
        if val_str.startswith('▼'): return 'color: #c0392b;' # Red
        return 'color: #7f8c8d;' # Grey

    df['vs. Their Avg.'] = df['delta'].apply(format_delta)
    
    st.dataframe(
        df.style.applymap(style_delta, subset=['vs. Their Avg.']),
        column_config={
            "critic_name": "Critic",
            "score": st.column_config.ProgressColumn("Their Score", format="%.1f", min_value=0, max_value=10),
            critic_avg_col: st.column_config.NumberColumn("Critic's Avg.", help="This critic's average score across all games.", format="%.2f"),
        },
        hide_index=True, use_container_width=True,
        column_order=['critic_name', 'score', critic_avg_col, 'vs. Their Avg.']
    )
    st.markdown(f"""<small><b>Legend:</b><br>
        <b>▲ Higher / ▼ Lower</b>: Score is more than {threshold:.2f} points different from the critic's personal average.<br>
        <b>~ About the Same</b>: Score is within {threshold:.2f} points of their personal average.</small>""",
        unsafe_allow_html=True)

def display_skipped_critics(critics_with_stats_df, rated_critic_ids):
    """Displays the list of critics who did not rate the selected game."""
    unrated_critics_df = critics_with_stats_df[~critics_with_stats_df['id'].isin(rated_critic_ids)]
    cols_to_display = ['critic_name']
    
    # Check if a pessimistic score column exists to display it
    if 'pessimistic_prior' in unrated_critics_df.columns:
        cols_to_display.append('pessimistic_prior')

    st.dataframe(
        unrated_critics_df,
        column_config={
            "critic_name": "Critic Name",
            "pessimistic_prior": st.column_config.NumberColumn("Pessimistic Score", help="The score used for this critic in the adjusted ranking calculation.", format="%.2f")
        },
        use_container_width=True,
        hide_index=True,
        column_order=cols_to_display
    )

# --- Main Application ---
def main():
    # --- Authentication & Page Setup ---
    if not st.session_state.get("password_correct", False):
        st.error("You need to log in first.")
        st.stop()

    st.set_page_config(page_title="Game Details", layout="wide")
    st.markdown("""
    <style>
    /* Centers the metric value and delta */
    div[data-testid="stMetric"] {
        text-align: center;
    }

    /* Centers the metric label */
    div[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Game Deep Dive")

    # --- Database Connection & Data Loading ---
    try:
        conn = st.connection("mydb", type="sql")
        games_df, critics_with_stats_df, ratings_df, rankings_df, global_stats = load_data(conn)
    except Exception as e:
        st.error(f"Database connection or data loading failed: {e}")
        st.stop()

    # --- Game Selection ---
    selected_game_name = st.selectbox(
        "Select a Game to Analyze:",
        options=games_df['game_name']
    )

    if not selected_game_name:
        st.info("Select a game to begin analysis.")
        st.stop()
        
    # --- Filter Data for Selected Game ---
    game_info = rankings_df.loc[rankings_df['game_name'] == selected_game_name].iloc[0]
    game_ratings_df = ratings_df[ratings_df['game_id'] == game_info['id']]
    
    if game_ratings_df.empty:
        st.warning(f"No ratings have been submitted for **{selected_game_name}** yet.")
        st.stop()
        
    # Merge ratings with critic info for use in multiple components
    game_ratings_with_critics = pd.merge(game_ratings_df, critics_with_stats_df, left_on='critic_id', right_on='id')

    # --- Render Page Components ---
    display_scorecard(game_info, game_ratings_df, len(critics_with_stats_df), global_stats)
    display_ranking_breakdown(game_info)

    st.subheader("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Critic Ratings", "Who Skipped?"])

    with tab1:
        display_score_distribution(game_ratings_with_critics)
    
    with tab2:
        display_critic_ratings(game_ratings_with_critics, global_stats['std_dev'])
        
    with tab3:
        display_skipped_critics(critics_with_stats_df, game_ratings_with_critics['critic_id'].unique())


if __name__ == "__main__":
    main()