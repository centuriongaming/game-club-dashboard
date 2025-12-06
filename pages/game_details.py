import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats
from utils import calculate_custom_game_rankings
import json
from datetime import datetime

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
    Loads game data, joined with enriched details (tags, price, metacritic),
    and integrates predictions.
    """
    # JOIN with games_details to get the new metadata
    games_query = """
    SELECT 
        g.id, 
        g.game_name, 
        gd.metacritic_score, 
        gd.price_usd, 
        gd.release_date, 
        gd.user_tags, 
        gd.developers, 
        gd.publishers
    FROM games g
    LEFT JOIN games_details gd ON g.id = gd.id
    WHERE g.upcoming IS FALSE 
    ORDER BY g.game_name;
    """
    games_df = _conn.query(games_query)
    
    critics_df = _conn.query("SELECT id, critic_name FROM critics ORDER BY critic_name;")
    ratings_df = _conn.query("SELECT critic_id, game_id, score FROM ratings;").dropna(subset=['score']).copy()
    preds_df = _conn.query("SELECT critic_id, id as game_id, predicted_score, predicted_skip_probability FROM critic_predictions;")

    rankings_df, critics_with_stats_df = calculate_custom_game_rankings(games_df, critics_df, ratings_df.copy())
    
    global_stats = {
        'avg_score': ratings_df['score'].mean(),
        'std_dev': ratings_df['score'].std(),
        'adjusted_avg': rankings_df['final_adjusted_score'].mean(),
        'adjusted_std': rankings_df['final_adjusted_score'].std(), 
    }
    
    return games_df, critics_with_stats_df, ratings_df, rankings_df, preds_df, global_stats

# --- Helpers ---
def calculate_age(release_date_str):
    if pd.isna(release_date_str) or release_date_str == 'N/A':
        return "Unknown"
    try:
        # Flexible parsing for various date formats
        dt = pd.to_datetime(release_date_str, format='mixed')
        years = (datetime.now() - dt).days / 365.25
        if years < 1:
            months = (datetime.now() - dt).days / 30
            return f"{int(months)} months old"
        return f"{years:.1f} years old"
    except:
        return release_date_str

def parse_tags(tags_str):
    if pd.isna(tags_str): return []
    try:
        # If it's already a list (from JSONB column), return it
        if isinstance(tags_str, list): return tags_str
        # If it's a string, parse it
        return json.loads(tags_str)
    except:
        return []

# --- UI Component Functions ---

def display_game_metadata(row):
    """
    Displays the new rich metadata (Price, Age, Metacritic, Tags) in a visual container.
    """
    st.caption(f"Developed by **{row['developers'] or 'Unknown'}** • Published by **{row['publishers'] or 'Unknown'}**")
    
    # 1. Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        price = row['price_usd']
        val = "Free" if price == 0 else f"${price:.2f}"
        st.metric("Price", val)
        
    with m2:
        age = calculate_age(row['release_date'])
        st.metric("Age", age, help=f"Released: {row['release_date']}")
        
    with m3:
        meta = row['metacritic_score']
        val = meta if (pd.notnull(meta) and meta != 'N/A') else "N/A"
        st.metric("Metacritic", val)

    with m4:
        # Simple genre placeholder or extra stat
        count = len(parse_tags(row['user_tags']))
        st.metric("Tags", f"{count}", help="Total Steam tags associated with this game.")

    # 2. Tag Chips
    tags = parse_tags(row['user_tags'])
    if tags:
        # CSS to make them look like chips/badges
        chip_style = """
        <span style='
            background-color: #31333F; 
            color: #FFFFFF; 
            border-radius: 15px; 
            padding: 5px 12px; 
            margin: 0 5px 10px 0; 
            font-size: 0.85em; 
            display: inline-block;
            border: 1px solid #4B4B4B;
        '>
        """
        
        # Taking top 12 tags to avoid clutter
        html_tags = "".join([f"{chip_style}{t}</span>" for t in tags[:12]])
        st.markdown(f"<div style='margin-top: 10px; margin-bottom: 20px;'>{html_tags}</div>", unsafe_allow_html=True)

def create_gauge_figure(value, reference, title):
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
    st.subheader("Performance Scorecard")
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.plotly_chart(create_gauge_figure(game_info['average_score'], global_stats['avg_score'], "Raw Average Score"), use_container_width=True)
                st.metric("Unadjusted Rank", f"#{int(game_info['Unadjusted Rank'])}")
            with g_col2:
                st.plotly_chart(create_gauge_figure(game_info['final_adjusted_score'], global_stats['adjusted_avg'], "Final Adjusted Score"), use_container_width=True)
                st.metric("Adjusted Rank", f"#{int(game_info['Rank'])}")
            st.caption("The ▲/▼ number below each score shows its difference from the average of all games.")

        with col2:
            st.metric("Number of Ratings", f"{game_info['number_of_ratings']}")
            play_rate = (game_info['number_of_ratings'] / num_total_critics) * 100
            st.metric("Play Rate", f"{play_rate:.1f}%")
            std_dev = game_ratings_df['score'].std() if game_info['number_of_ratings'] > 1 else "N/A"
            st.metric("Controversy (Std Dev)", f"{std_dev:.3f}" if isinstance(std_dev, float) else std_dev)

def display_ranking_breakdown(game_info, critics_df, skipper_ids):
    st.subheader("Ranking Breakdown")
    with st.expander("How the Final Adjusted Score is Calculated", expanded=False):
        st.markdown("##### Pessimistic Prior Calculation")
        st.markdown("The prior is the average of the personal 'pessimistic scores' from each critic who skipped this game.")

        skipper_df = critics_df[critics_df['id'].isin(skipper_ids)].copy()
        skipper_df['critic_avg'] = skipper_df['critic_avg'].fillna(0)
        skipper_df['critic_std'] = skipper_df['critic_std'].fillna(0)
        skipper_df['pessimistic_score'] = skipper_df['critic_avg'] - skipper_df['critic_std']

        st.dataframe(
            skipper_df[['critic_name', 'critic_avg', 'critic_std', 'pessimistic_score']],
            column_config={
                "critic_name": "Skipping Critic",
                "critic_avg": st.column_config.NumberColumn("Their Avg", format="%.2f"),
                "critic_std": st.column_config.NumberColumn("Their Std Dev", format="%.2f"),
                "pessimistic_score": st.column_config.NumberColumn("Pessimistic Score", format="%.3f")
            },
            use_container_width=True, hide_index=True
        )

        sum_of_priors = skipper_df['pessimistic_score'].sum()
        num_skippers = len(skipper_df)
        final_prior = game_info['pessimistic_prior']
        
        st.markdown(fr'$$ \text{{Avg Pessimistic Score}} = \frac{{{sum_of_priors:.3f}}}{{{num_skippers}}} = \textbf{{{final_prior:.3f}}} $$')
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
    scores = game_ratings_with_critics['score']
    if len(scores) > 1 and scores.nunique() > 1:
        kde = stats.gaussian_kde(scores)
        x_range = np.linspace(0, 10, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', fill='tozeroy', line_shape='spline', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=scores, y=[0.005] * len(scores), mode='markers', marker=dict(symbol='line-ns-open', color='black', size=10)))
        fig.update_layout(showlegend=False, xaxis_title="Score", yaxis_title="Density", xaxis=dict(range=[0, 10]), margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    elif len(scores) > 1:
        st.info("Distribution plot unavailable (All ratings are identical).")
    else:
        st.info("Need at least 2 ratings for distribution plot.")

def display_critic_ratings(game_ratings_with_critics, global_std_dev):
    df = game_ratings_with_critics.copy()
    if 'predicted_score' in df.columns:
        df['surprise'] = df['score'] - df['predicted_score']
    else:
        df['surprise'] = np.nan

    def format_surprise(val):
        if pd.isna(val): return "N/A"
        if val > 1.0: return f"▲ +{val:.1f} (Loved it)"
        if val < -1.0: return f"▼ {val:.1f} (Let down)"
        return f"~ {val:+.1f} (Expected)"

    def style_surprise(val_str):
        if '▲' in val_str: return 'color: #2ecc71; font-weight: bold;' 
        if '▼' in val_str: return 'color: #e74c3c; font-weight: bold;'
        return 'color: #95a5a6;'

    df['Surprise Factor'] = df['surprise'].apply(format_surprise)
    
    st.dataframe(
        df.style.applymap(style_surprise, subset=['Surprise Factor']),
        column_config={
            "critic_name": "Critic",
            "score": st.column_config.NumberColumn("Actual", format="%.1f"),
            "predicted_score": st.column_config.NumberColumn("Predicted", format="%.1f"),
            "Surprise Factor": st.column_config.Column("Surprise")
        },
        hide_index=True, use_container_width=True,
        column_order=['critic_name', 'score', 'predicted_score', 'Surprise Factor']
    )

def display_skipped_critics(critics_with_stats_df, rated_critic_ids, preds_df, game_id):
    unrated_critics = critics_with_stats_df[~critics_with_stats_df['id'].isin(rated_critic_ids)].copy()
    game_preds = preds_df[preds_df['game_id'] == game_id]
    merged_df = pd.merge(unrated_critics, game_preds, left_on='id', right_on='critic_id', how='left')
    merged_df['skip_prob_pct'] = merged_df['predicted_skip_probability'] * 100
    
    st.dataframe(
        merged_df,
        column_config={
            "critic_name": "Critic Name",
            "skip_prob_pct": st.column_config.ProgressColumn("Skip Probability", format="%.0f%%", min_value=0, max_value=100),
            "predicted_score": st.column_config.NumberColumn("Hypothetical Score", format="%.1f"),
            "pessimistic_prior": st.column_config.NumberColumn("Ranking Penalty", format="%.2f")
        },
        use_container_width=True, hide_index=True,
        column_order=['critic_name', 'skip_prob_pct', 'predicted_score', 'pessimistic_prior']
    )

# --- Main Application ---
def main():
    if not st.session_state.get("password_correct", False):
        st.error("You need to log in first.")
        st.stop()

    st.set_page_config(page_title="Game Details", layout="wide")
    st.markdown("""<style>div[data-testid="stMetric"] { display: flex; flex-direction: column; align-items: center; } div[data-testid="stMetricValue"] { text-align: center; }</style>""", unsafe_allow_html=True)
    st.title("Game Details")

    try:
        conn = st.connection("mydb", type="sql")
        games_df, critics_with_stats_df, ratings_df, rankings_df, preds_df, global_stats = load_data(conn)
    except Exception as e:
        st.error(f"Database connection or data loading failed: {e}")
        st.stop()

    selected_game_name = st.selectbox("Select a Game to Analyze:", options=games_df['game_name'])
    if not selected_game_name:
        st.stop()

    game_info = rankings_df.loc[rankings_df['game_name'] == selected_game_name].iloc[0]
    game_id = game_info['id']
    game_ratings_df = ratings_df[ratings_df['game_id'] == game_id]
    
    # NEW: Display the Metadata
    display_game_metadata(game_info)

    if game_ratings_df.empty:
        st.warning(f"No ratings for {selected_game_name}.")
        st.stop()
        
    game_ratings_with_critics = pd.merge(game_ratings_df, critics_with_stats_df, left_on='critic_id', right_on='id')
    game_ratings_with_critics = pd.merge(game_ratings_with_critics, preds_df[preds_df['game_id'] == game_id], on='critic_id', how='left')

    display_scorecard(game_info, game_ratings_df, len(critics_with_stats_df), global_stats)
    
    rated_critic_ids = set(game_ratings_with_critics['critic_id'])
    display_ranking_breakdown(game_info, critics_with_stats_df, set(critics_with_stats_df['id']) - rated_critic_ids)

    st.subheader("Detailed Analysis")
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Critic Ratings vs Predictions", "Who Skipped & Why"])
    with tab1: display_score_distribution(game_ratings_with_critics)
    with tab2: display_critic_ratings(game_ratings_with_critics, global_stats['std_dev'])
    with tab3: display_skipped_critics(critics_with_stats_df, rated_critic_ids, preds_df, game_id)

if __name__ == "__main__":
    main()