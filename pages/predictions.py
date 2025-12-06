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

# --- Constants for Styling ---
COLOR_POS = "#1f77b4" # Muted Blue (Colorblind Safe)
COLOR_NEG = "#ff7f0e" # Safety Orange (Colorblind Safe)

# --- Feature Engineering Helpers ---
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
    features = []
    tags = clean_tags(row.get('user_tags'))
    features.extend([f"tag__{t}" for t in tags])
    features.append(f"bin__{bin_price(row.get('price_usd'))}")
    features.append(f"bin__{bin_release_date_dynamic(row.get('release_date'))}")
    return features

# --- Data Loading ---
@st.cache_data
def load_prediction_data(_session):
    # 1. Base Tables
    critics_df = pd.read_sql(sa.select(Critic.id.label('critic_id'), Critic.critic_name).order_by(Critic.critic_name), _session.bind)
    
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
    
    importances_df = pd.read_sql(
        sa.select(CriticFeatureImportance.feature, CriticFeatureImportance.importance, Critic.critic_name, CriticFeatureImportance.model_type)
        .join(Critic, Critic.id == CriticFeatureImportance.critic_id)
        .where(CriticFeatureImportance.model_type == 'deviation_weighted'),
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

# --- Helper: Diverging Bar Chart ---
def plot_diverging_bar(data, x_col, y_col, title, height=400):
    """Creates a center-aligned diverging bar chart (Blue positive, Orange negative)."""
    data = data.copy()
    # Add sentiment column for coloring
    data['Sentiment'] = data[x_col].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    
    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col, 
        orientation='h',
        title=title,
        text_auto='.2f',
        color='Sentiment',
        color_discrete_map={'Positive': COLOR_POS, 'Negative': COLOR_NEG}
    )
    
    fig.update_layout(
        showlegend=False,
        xaxis_title="Relative Affinity (Right = Loves, Left = Dislikes)",
        yaxis_title=None,
        height=height,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='white'),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_yaxes(categoryorder='total ascending')
    return fig

# --- UI Components ---

def display_critic_profile(importances_df, selected_critic):
    st.subheader(f"🧠 Taste Profile: {selected_critic}")
    
    critic_data = importances_df[importances_df['critic_name'] == selected_critic].copy()
    if critic_data.empty:
        st.info("No affinity profile data found.")
        return

    # 1. Price Sensitivity
    price_df = critic_data[critic_data['feature'].str.startswith('bin__Price')].copy()
    price_df['Label'] = price_df['feature'].str.replace('bin__Price_', '').str.replace('_', ' ')
    price_order = ["Free", "$0.01-$14.99", "$15.00-$29.99", "$30.00-$49.99", "$50.00+"]
    
    # 2. Age Sensitivity
    age_df = critic_data[critic_data['feature'].str.startswith('bin__Age')].copy()
    age_df['Label'] = age_df['feature'].str.replace('bin__Age_', '').str.replace('_', ' ')
    age_order = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]

    # 3. Tags (Merged Diverging Chart)
    tag_df = critic_data[critic_data['feature'].str.startswith('tag__')].copy()
    tag_df['Label'] = tag_df['feature'].str.replace('tag__', '')
    
    # Find most impactful tags (absolute value)
    tag_df['abs_importance'] = tag_df['importance'].abs()
    top_tags = tag_df.sort_values('abs_importance', ascending=False).head(12)

    # Render Layout
    c1, c2 = st.columns(2)
    with c1:
        # Price Chart (Diverging)
        fig_price = plot_diverging_bar(price_df, 'importance', 'Label', "Price Sensitivity", height=300)
        fig_price.update_yaxes(categoryorder='array', categoryarray=price_order)
        st.plotly_chart(fig_price, use_container_width=True)
    with c2:
        # Age Chart (Diverging)
        fig_age = plot_diverging_bar(age_df, 'importance', 'Label', "Age Preference", height=300)
        fig_age.update_yaxes(categoryorder='array', categoryarray=age_order)
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("#### Most Polarizing Genres & Tags")
    st.caption("Tags pointing **Right (Blue)** are favorites. Tags pointing **Left (Orange)** are disliked.")
    st.plotly_chart(plot_diverging_bar(top_tags, 'importance', 'Label', "", height=500), use_container_width=True)


def display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game):
    st.markdown(f"### Prediction Analysis: {selected_game}")
    
    # 1. Get Prediction Record
    record = merged_df[(merged_df['critic_name'] == selected_critic) & (merged_df['game_name'] == selected_game)]
    if record.empty:
        st.warning("No data.")
        return
    record = record.iloc[0]
    
    is_upcoming = record['upcoming']

    # 2. Get Game Features
    game_meta = games_df[games_df['game_name'] == selected_game].iloc[0]
    game_features = get_game_features(game_meta)

    # 3. Match with Critic Profile
    critic_profile = importances_df[importances_df['critic_name'] == selected_critic]
    
    contributions = []
    for f in game_features:
        match = critic_profile[critic_profile['feature'] == f]
        affinity = match.iloc[0]['importance'] if not match.empty else 0.0
        clean_name = f.replace('tag__', '').replace('bin__', '').replace('_', ' ')
        contributions.append({'Feature': clean_name, 'Impact': affinity})
    
    contrib_df = pd.DataFrame(contributions)
    
    # 4. Truncate for Readability
    contrib_df['abs_impact'] = contrib_df['Impact'].abs()
    contrib_df = contrib_df.sort_values('abs_impact', ascending=False).head(12)

    # --- Render UI ---
    
    # Top Row: The Numbers
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        
        # Metric 1: Predicted Score
        pred_val = f"{record['predicted_score']:.2f}" if pd.notna(record['predicted_score']) else "N/A"
        c1.metric("Predicted Score", pred_val, help="The score the model thinks this critic will give based on their taste profile.")
        
        # Metric 2: Skip Prob
        skip_val = f"{record['predicted_skip_probability']*100:.1f}%" if pd.notna(record['predicted_skip_probability']) else "N/A"
        c2.metric("Skip Probability", skip_val, help="The likelihood that the critic will choose NOT to play this game.")
        
        # Metric 3: Actual Outcome (Handles Upcoming)
        if is_upcoming:
            actual_val = "Upcoming"
            actual_help = "This game has not been released yet."
        else:
            if pd.notna(record['score']):
                actual_val = f"{record['score']:.1f}"
                actual_help = "The actual score given by the critic."
            elif record['actual_skip']:
                actual_val = "Skipped"
                actual_help = "The critic chose not to play this game."
            else:
                actual_val = "Pending"
                actual_help = "Released, but not yet rated."
                
        c3.metric("Actual Outcome", actual_val, help=actual_help)

    # Bottom Row: The "Why"
    st.markdown("#### Key Drivers")
    st.caption("How the specific features of this game matched the critic's profile.")
    
    if not contrib_df.empty:
        st.plotly_chart(plot_diverging_bar(contrib_df, 'Impact', 'Feature', ""), use_container_width=True)
    else:
        st.info("No feature data available for this game.")

# --- Main ---
def main():
    check_auth()
    session = get_sqla_session()
    
    st.title("Predictive Analytics")
    
    # Load Data
    critic_names, games_df, merged_df, importances_df = load_prediction_data(session)
    game_names = games_df['game_name'].tolist()

    if merged_df.empty:
        st.error("No data available.")
        st.stop()

    # 1. Critic Selection (Top)
    selected_critic = st.selectbox("Select Critic", critic_names, index=0)
    
    # 2. Profile View (Middle)
    display_critic_profile(importances_df, selected_critic)
    
    st.divider()
    
    # 3. Game Selection (Bottom Section)
    st.subheader("Game Prediction Breakdown")
    selected_game = st.selectbox("Select Game to Analyze", game_names, index=0)
    
    # 4. Breakdown
    display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game)
    
    # Footer
    st.write("")
    st.write("")
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()