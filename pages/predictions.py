# pages/predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy as sa
import plotly.express as px
import plotly.graph_objects as go  # Added for Waterfall support if needed later
import json
import ast
from datetime import datetime
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating, CriticPrediction, CriticFeatureImportance, GameDetails

# --- Page Configuration ---
st.set_page_config(page_title="Predictions", layout="wide")

# --- Constants for Styling ---
COLOR_POS = "#2E86C1" # Strong Blue (Clear Positive)
COLOR_NEG = "#E74C3C" # Soft Red (Clear Negative) - Changed from Orange for better "Stop/Go" semantic

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

# --- Visualization Helper ---
def plot_diverging_bar(data, x_col, y_col, title, height=400, x_label="Impact on Score (Points)"):
    """
    Standardized Diverging Bar Chart.
    Best for: Comparing 'Pros' vs 'Cons' relative to a neutral baseline.
    """
    data = data.copy()
    data['Sentiment'] = data[x_col].apply(lambda x: 'Increases Score' if x >= 0 else 'Decreases Score')
    data['Tooltip'] = data[x_col].apply(lambda x: f"{x:+.2f} pts")

    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col, 
        orientation='h',
        title=title,
        text='Tooltip',
        color='Sentiment',
        color_discrete_map={'Increases Score': COLOR_POS, 'Decreases Score': COLOR_NEG},
        hover_data={'Sentiment': True, x_col: False, y_col: False}
    )
    
    # Visual Polish: Add a Zero Line
    fig.add_vline(x=0, line_width=2, line_color="white", opacity=0.5)

    fig.update_layout(
        showlegend=True,
        legend_title=None,
        xaxis_title=x_label,
        yaxis_title=None,
        height=height,
        xaxis=dict(zeroline=False, gridcolor='#333'), # Disable default zero line to use our custom one
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=14)
    )
    fig.update_yaxes(categoryorder='total ascending')
    return fig

# --- Explainer ---
def render_algorithm_explainer():
    with st.expander("ℹ️ How does this work? (Click to Expand)"):
        st.markdown("""
        ### 🤖 Inside the Prediction Engine
        This system predicts enjoyment using **Deviation Analysis**.
        
        1.  **The Baseline:** Every critic has a "Baseline Average" (e.g., 7.5/10). This is their starting point.
        2.  **The Adjustments:** We look at the game's features (Genre, Price, Age).
            * If the critic *loves* RPGs, we might add **+0.5 points**.
            * If they *hate* Expensive games, we might subtract **-0.8 points**.
        3.  **The Prediction:** `Baseline + Adjustments = Final Score`.
        
        *Note: We also learn from **Skipped Games**. If a critic chooses not to play a game, the system treats that as a negative signal for those specific features.*
        """)

# --- UI Components ---

# --- Updated Profile View ---
def display_critic_profile(importances_df, selected_critic, critic_baseline):
    st.subheader(f"🧠 Taste Profile: {selected_critic}")
    
    # Metric Row
    c_base, c_count, c_top = st.columns(3)
    c_base.metric("Baseline Score", f"{critic_baseline:.2f}", help="The average score this critic gives.")
    
    critic_data = importances_df[importances_df['critic_name'] == selected_critic].copy()
    if critic_data.empty:
        st.warning("No profile data found.")
        return
    
    # Filter for Tags only (remove price/age bins for the main tag view)
    tag_df = critic_data[critic_data['feature'].str.startswith('tag__')].copy()
    tag_df['feature'] = tag_df['feature'].str.replace('tag__', '') # Clean names
    
    c_count.metric("Learned Tags", len(tag_df), help="Total number of tags we have learned preferences for.")
    
    # Find their #1 Obsession
    top_tag = tag_df.loc[tag_df['importance'].abs().idxmax()]
    c_top.metric(f"Top Driver", top_tag['feature'], f"{top_tag['importance']:+.2f} pts")

    st.divider()

    # --- 1. The Full Spectrum Treemap ---
    st.markdown("### 🗺️ The Taste Map (All Tags)")
    st.caption("This map shows every tag we've learned about. **Bigger Box = Stronger Opinion**. Blue is Positive, Orange is Negative.")
    
    treemap_fig = plot_taste_treemap(tag_df, f"{selected_critic}'s Complete Taste Profile")
    st.plotly_chart(treemap_fig, use_container_width=True)
    

    st.divider()

    # --- 2. Traditional Breakdowns (Price/Age) ---
    st.markdown("### 💰 & 📅 Price and Age Preferences")
    
    price_df = critic_data[critic_data['feature'].str.startswith('bin__Price')].copy()
    price_df['Label'] = price_df['feature'].str.replace('bin__Price_', '').str.replace('_', ' ')
    price_order = ["Free", "$0.01-$14.99", "$15.00-$29.99", "$30.00-$49.99", "$50.00+"]
    
    age_df = critic_data[critic_data['feature'].str.startswith('bin__Age')].copy()
    age_df['Label'] = age_df['feature'].str.replace('bin__Age_', '').str.replace('_', ' ')
    age_order = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]

    c1, c2 = st.columns(2)
    with c1:
        fig_price = plot_diverging_bar(price_df, 'importance', 'Label', "Price Sensitivity", height=300)
        fig_price.update_yaxes(categoryorder='array', categoryarray=price_order)
        st.plotly_chart(fig_price, use_container_width=True)
    with c2:
        fig_age = plot_diverging_bar(age_df, 'importance', 'Label', "Age Preference", height=300)
        fig_age.update_yaxes(categoryorder='array', categoryarray=age_order)
        st.plotly_chart(fig_age, use_container_width=True)

    # --- 3. Searchable Data Grid ---
    with st.expander("🔎 View Raw Data Grid (Searchable)"):
        st.markdown("Search for specific genres (e.g., 'Roguelike', 'FPS') to see exact scores.")
        
        # Format for display
        display_df = tag_df[['feature', 'importance']].copy()
        display_df.columns = ['Tag', 'Impact (Points)']
        display_df['Sentiment'] = display_df['Impact (Points)'].apply(lambda x: '❤️ Likes' if x > 0 else '💔 Dislikes')
        
        st.dataframe(
            display_df.sort_values('Impact (Points)', ascending=False),
            column_config={
                "Impact (Points)": st.column_config.NumberColumn(format="%.2f"),
            },
            use_container_width=True,
            hide_index=True
        )

def display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game, critic_baseline):
    st.markdown(f"### 🎯 Analysis: {selected_game}")
    
    record = merged_df[(merged_df['critic_name'] == selected_critic) & (merged_df['game_name'] == selected_game)]
    if record.empty:
        st.warning("No data found.")
        return
    record = record.iloc[0]
    
    predicted_score = record['predicted_score']
    
    # --- The Equation Visualizer ---
    st.markdown("#### 🧮 The Calculation")
    
    col_base, col_plus, col_adj, col_equals, col_final = st.columns([2, 0.5, 2, 0.5, 2])
    
    with col_base:
        st.metric("Critic Baseline", f"{critic_baseline:.2f}", help="The critic's average score.")
    with col_plus:
        st.markdown("<h3 style='text-align: center; color: gray;'>+</h3>", unsafe_allow_html=True)
    with col_adj:
        adj_val = predicted_score - critic_baseline if pd.notna(predicted_score) else 0.0
        st.metric("Total Adjustments", f"{adj_val:+.2f}", delta=adj_val, help="Sum of all feature impacts below.")
    with col_equals:
        st.markdown("<h3 style='text-align: center; color: gray;'>=</h3>", unsafe_allow_html=True)
    with col_final:
        st.metric("Predicted Score", f"{predicted_score:.2f}" if pd.notna(predicted_score) else "N/A")

    st.divider()

    # --- Feature Contributions ---
    game_meta = games_df[games_df['game_name'] == selected_game].iloc[0]
    game_features = get_game_features(game_meta)
    critic_profile = importances_df[importances_df['critic_name'] == selected_critic]
    
    contributions = []
    for f in game_features:
        match = critic_profile[critic_profile['feature'] == f]
        if not match.empty:
            affinity = match.iloc[0]['importance']
            clean_name = f.replace('tag__', '').replace('bin__', '').replace('_', ' ')
            contributions.append({'Feature': clean_name, 'Impact': affinity})
    
    contrib_df = pd.DataFrame(contributions)
    if not contrib_df.empty:
        contrib_df['abs_impact'] = contrib_df['Impact'].abs()
        contrib_df = contrib_df.sort_values('abs_impact', ascending=False).head(12)

    st.markdown("#### 🔍 What drove this score?")
    st.caption(f"These are the specific features of *{selected_game}* that {selected_critic} reacts to.")
    
    if not contrib_df.empty:
        st.plotly_chart(plot_diverging_bar(contrib_df, 'Impact', 'Feature', "", x_label="Impact on Score (Points)"), use_container_width=True)
    else:
        st.info("This game has no distinct features that match the critic's profile strong enough to sway the score.")
# --- New Helper: Taste Treemap ---
def plot_taste_treemap(data, title):
    """
    Generates a Treemap to show ALL tags at once.
    Hierarchy: Root -> Sentiment (Like/Dislike) -> Tag
    Size: Absolute Impact
    Color: Real Impact
    """
    data = data.copy()
    
    # 1. Create Hierarchy
    data['Category'] = data['importance'].apply(lambda x: '✅ Likes / Boosts Score' if x > 0 else '❌ Dislikes / Lowers Score')
    data['Abs_Impact'] = data['importance'].abs()
    
    # 2. Tooltip formatting
    data['Tooltip'] = data['importance'].apply(lambda x: f"{x:+.2f} points")

    # 3. Build Plot
    fig = px.treemap(
        data,
        path=[px.Constant("All Features"), 'Category', 'feature'], # The hierarchy
        values='Abs_Impact', # Size of the box
        color='importance',  # Color of the box
        color_continuous_scale=[(0, COLOR_NEG), (0.5, '#ffffff'), (1, COLOR_POS)], # Diverging scale
        color_continuous_midpoint=0,
        custom_data=['Tooltip', 'importance']
    )

    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Impact: %{customdata[0]}<extra></extra>',
        textinfo="label+value", # Show label and the size value
        texttemplate="<b>%{label}</b>", # Just show the name cleanly
    )

    fig.update_layout(
        title=title,
        height=600, # Taller for more detail
        margin=dict(t=50, l=10, r=10, b=10),
        coloraxis_showscale=False # Hide the color bar (the categories explain it enough)
    )
    
    return fig

# --- Main ---
def main():
    check_auth()
    session = get_sqla_session()
    
    st.title("🔮 Predictive Analytics")
    render_algorithm_explainer()
    
    critic_names, games_df, merged_df, importances_df = load_prediction_data(session)
    game_names = games_df['game_name'].tolist()

    if merged_df.empty:
        st.error("No data available.")
        st.stop()

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_critic = st.selectbox("Select Critic", critic_names, index=0)
    
    critic_rows = merged_df[merged_df['critic_name'] == selected_critic]
    actual_scores = critic_rows[critic_rows['score'].notna()]['score']
    critic_baseline = actual_scores.mean() if not actual_scores.empty else 5.0

    tab_profile, tab_game = st.tabs(["👤 Critic Profile", "🎲 Game Prediction Analysis"])
    
    with tab_profile:
        display_critic_profile(importances_df, selected_critic, critic_baseline)
        
    with tab_game:
        selected_game = st.selectbox("Select Game to Analyze", game_names, index=0)
        display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game, critic_baseline)
    
    st.write("")
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()