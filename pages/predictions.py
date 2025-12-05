# pages/predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy as sa
import plotly.express as px
import json
import ast
from datetime import datetime
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating, CriticPrediction, CriticFeatureImportance, GameDetails

# --- Page Configuration ---
st.set_page_config(page_title="Predictions", layout="wide")

# --- Feature Engineering Helpers (Must match master_updater.py) ---
def clean_tags(val):
    tags = []
    if isinstance(val, list): tags = val
    elif isinstance(val, str):
        try:
            if val.strip().startswith('['): tags = json.loads(val)
            else: tags = ast.literal_eval(val)
        except: tags = []
    return [str(t).strip().replace(' ', '_').replace('-', '_') for t in tags if isinstance(tags, list)]

def bin_price(price):
    if pd.isna(price): return "Price_Unknown"
    if price == 0: return "Price_Free"
    if price < 15: return "Price_$0.01-$14.99"
    if price < 30: return "Price_$15.00-$29.99"
    if price < 50: return "Price_$30.00-$49.99"
    return "Price_$50.00+"

def bin_release_date_dynamic(release_date_str):
    if pd.isna(release_date_str) or release_date_str == 'N/A': return "Age_Unknown"
    try:
        dt = pd.to_datetime(release_date_str, format='mixed')
        age_days = (datetime.now() - dt).days
        years = age_days / 365.25
        if years < 1: return "Age_<1y"
        if years < 3: return "Age_1-3y"
        if years < 5: return "Age_3-5y"
        if years < 10: return "Age_5-10y"
        return "Age_10y+"
    except: return "Age_Unknown"

def get_game_features(row):
    """Reconstructs the feature list for a single game row."""
    features = []
    # Tags
    tags = clean_tags(row.get('user_tags'))
    features.extend([f"tag__{t}" for t in tags])
    # Price
    features.append(f"bin__{bin_price(row.get('price_usd'))}")
    # Age
    features.append(f"bin__{bin_release_date_dynamic(row.get('release_date'))}")
    return features

# --- Data Loading ---
@st.cache_data
def load_prediction_data(_session):
    """Loads critics, games, details, predictions, and affinities."""
    # 1. Base Tables
    critics_df = pd.read_sql(sa.select(Critic.id.label('critic_id'), Critic.critic_name).order_by(Critic.critic_name), _session.bind)
    
    # Join Game + GameDetails to get metadata needed for features
    games_query = sa.select(
        Game.id.label('game_id'), 
        Game.game_name, 
        Game.upcoming,
        GameDetails.price_usd,
        GameDetails.release_date,
        GameDetails.user_tags
    ).join(GameDetails, Game.id == GameDetails.id, isouter=True).order_by(Game.game_name)
    
    games_df = pd.read_sql(games_query, _session.bind)
    
    predictions_df = pd.read_sql(sa.select(CriticPrediction.critic_id, CriticPrediction.id.label('game_id'), CriticPrediction.predicted_score, CriticPrediction.predicted_skip_probability), _session.bind)
    ratings_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), _session.bind)
    
    # Fetch Affinities (CriticFeatureImportance)
    # We filter for 'relative_affinity' as that is our new main model type
    importances_df = pd.read_sql(
        sa.select(CriticFeatureImportance.feature, CriticFeatureImportance.importance, Critic.critic_name, CriticFeatureImportance.model_type)
        .join(Critic, Critic.id == CriticFeatureImportance.critic_id)
        .where(CriticFeatureImportance.model_type == 'relative_affinity'),
        _session.bind
    )

    # 2. Merge Prediction Data
    scaffold_df = pd.MultiIndex.from_product(
        [critics_df['critic_id'], games_df['game_id']],
        names=['critic_id', 'game_id']
    ).to_frame(index=False)

    merged_df = pd.merge(scaffold_df, critics_df, on='critic_id', how='left')
    merged_df = pd.merge(merged_df, games_df, on='game_id', how='left')
    merged_df = pd.merge(merged_df, predictions_df, on=['critic_id', 'game_id'], how='left')
    merged_df = pd.merge(merged_df, ratings_df, on=['critic_id', 'game_id'], how='left')
    merged_df['actual_skip'] = merged_df['score'].isna()
    
    return critics_df['critic_name'].tolist(), games_df, merged_df, importances_df

# --- Visualization Components ---

def display_critic_profile(importances_df, selected_critic):
    """Displays Price, Age, and Tag affinities."""
    st.subheader(f"Affinity Profile: {selected_critic}")
    
    critic_data = importances_df[importances_df['critic_name'] == selected_critic].copy()
    if critic_data.empty:
        st.info("No affinity profile data found.")
        return

    # Helper to plot nicely
    def plot_affinity_bar(data, x_col, y_col, title, color_discrete_sequence=None, category_orders=None):
        fig = px.bar(data, x=x_col, y=y_col, title=title, text_auto='.2f', color=y_col, 
                     color_continuous_scale=px.colors.diverging.RdBu)
        fig.update_layout(coloraxis_showscale=False, yaxis_title=None, xaxis_title="Relative Affinity (Higher = Better)")
        if category_orders:
            fig.update_xaxes(categoryorder='array', categoryarray=category_orders)
        return fig

    # 1. Price Sensitivity
    price_df = critic_data[critic_data['feature'].str.startswith('bin__Price')].copy()
    price_df['Label'] = price_df['feature'].str.replace('bin__Price_', '').str.replace('_', ' ')
    # Define logical sort order
    price_order = ["Free", "$0.01-$14.99", "$15.00-$29.99", "$30.00-$49.99", "$50.00+"]
    
    # 2. Age Sensitivity
    age_df = critic_data[critic_data['feature'].str.startswith('bin__Age')].copy()
    age_df['Label'] = age_df['feature'].str.replace('bin__Age_', '').str.replace('_', ' ')
    age_order = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]

    # 3. Tags (Top/Bottom)
    tag_df = critic_data[critic_data['feature'].str.startswith('tag__')].copy()
    tag_df['Label'] = tag_df['feature'].str.replace('tag__', '')
    tag_df = tag_df.sort_values('importance', ascending=False)
    
    top_5 = tag_df.head(5)
    bottom_5 = tag_df.tail(5).sort_values('importance', ascending=True) # Sort to show most negative first

    # Render Layout
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_affinity_bar(price_df, 'Label', 'importance', "Price Sensitivity", category_orders=price_order), use_container_width=True)
    with col2:
        st.plotly_chart(plot_affinity_bar(age_df, 'Label', 'importance', "Age Preference", category_orders=age_order), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_top = px.bar(top_5, x='importance', y='Label', orientation='h', title="Top 5 Loved Tags", text_auto='.2f')
        fig_top.update_traces(marker_color='#2E8B57') # SeaGreen
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top, use_container_width=True)
    with col4:
        fig_bot = px.bar(bottom_5, x='importance', y='Label', orientation='h', title="Top 5 Disliked Tags", text_auto='.2f')
        fig_bot.update_traces(marker_color='#CD5C5C') # IndianRed
        fig_bot.update_layout(yaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_bot, use_container_width=True)


def display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game):
    """Shows the prediction result AND the factors that drove it."""
    
    # 1. Get Prediction Record
    record = merged_df[(merged_df['critic_name'] == selected_critic) & (merged_df['game_name'] == selected_game)]
    if record.empty:
        st.warning("No data.")
        return
    record = record.iloc[0]

    # 2. Get Game Metadata & Features
    game_meta = games_df[games_df['game_name'] == selected_game].iloc[0]
    # Re-calculate features on the fly using the helper functions
    game_features = get_game_features(game_meta)

    # 3. Get Critic Profile
    critic_profile = importances_df[importances_df['critic_name'] == selected_critic]
    
    # 4. Match Features
    # Create a DataFrame of the features IN THIS GAME and the critic's affinity for them
    contributions = []
    for f in game_features:
        # Find affinity in critic profile
        match = critic_profile[critic_profile['feature'] == f]
        affinity = match.iloc[0]['importance'] if not match.empty else 0.0 # Default to neutral if missing
        
        # formatting for display
        clean_name = f.replace('tag__', 'Tag: ').replace('bin__', '')
        contributions.append({'Feature': clean_name, 'Impact': affinity})
    
    contrib_df = pd.DataFrame(contributions).sort_values('Impact', ascending=False)

    # --- Render UI ---
    st.markdown(f"### Prediction Analysis: {selected_game}")
    
    # Top Row: The Numbers
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Score", f"{record['predicted_score']:.2f}" if pd.notna(record['predicted_score']) else "N/A")
        c2.metric("Skip Probability", f"{record['predicted_skip_probability']*100:.1f}%" if pd.notna(record['predicted_skip_probability']) else "N/A")
        
        actual_val = f"{record['score']:.1f}" if pd.notna(record['score']) else ("Skipped" if record['actual_skip'] else "Pending")
        c3.metric("Actual Outcome", actual_val)

    # Bottom Row: The "Why"
    st.markdown("#### What drove this prediction?")
    st.caption("These are the specific features of this game and how much this critic likes/dislikes them relative to the average.")
    
    if not contrib_df.empty:
        # Color code: Green for positive, Red for negative
        fig = px.bar(contrib_df, x='Impact', y='Feature', orientation='h', 
                     color='Impact', color_continuous_scale=px.colors.diverging.RdBu,
                     text_auto='.2f')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature data available for this game.")

# --- Main ---
def main():
    check_auth()
    session = get_sqla_session()
    
    critic_names, games_df, merged_df, importances_df = load_prediction_data(session)
    game_names = games_df['game_name'].tolist()

    if merged_df.empty:
        st.error("No data available.")
        st.stop()

    # Sidebar / Selection
    with st.sidebar:
        st.header("Settings")
        selected_critic = st.selectbox("Critic", critic_names)
        selected_game = st.selectbox("Game Analysis", game_names)
    
    # Main Content
    st.title(f"Predictive Insights: {selected_critic}")
    
    # 1. Profile View
    display_critic_profile(importances_df, selected_critic)
    
    st.divider()
    
    # 2. Single Game Breakdown
    display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game)

if __name__ == "__main__":
    main()