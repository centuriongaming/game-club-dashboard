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
COLOR_POS = "#2E86C1" 
COLOR_NEG = "#E74C3C" 

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
def plot_diverging_bar(data, x_col, y_col, title, height=400, x_label="Preference Delta (Points)"):
    if data.empty:
        return None

    data = data.copy()
    data['Sentiment'] = data[x_col].apply(lambda x: 'Positive Affinity' if x >= 0 else 'Negative Affinity')
    data['Tooltip'] = data[x_col].apply(lambda x: f"{x:+.2f}")

    fig = px.bar(
        data, 
        x=x_col, 
        y=y_col, 
        orientation='h',
        title=title,
        text='Tooltip',
        color='Sentiment',
        color_discrete_map={'Positive Affinity': COLOR_POS, 'Negative Affinity': COLOR_NEG},
        hover_data={'Sentiment': True, x_col: False, y_col: False}
    )
    
    fig.add_vline(x=0, line_width=2, line_color="#555", opacity=0.8)

    fig.update_layout(
        showlegend=True,
        legend_title=None,
        xaxis_title=x_label,
        yaxis_title=None,
        height=height,
        xaxis=dict(zeroline=False, gridcolor='#eee'), 
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(size=12, family="Arial")
    )
    fig.update_yaxes(categoryorder='total ascending')
    return fig

# --- Helper: Taste Treemap ---
def plot_taste_treemap(data, title):
    if data.empty:
        return None

    data = data.copy()
    data['Category'] = data['importance'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    data['Abs_Impact'] = data['importance'].abs()
    data['Tooltip'] = data['importance'].apply(lambda x: f"{x:+.2f}")

    fig = px.treemap(
        data,
        path=[px.Constant("All Features"), 'Category', 'feature'],
        values='Abs_Impact',
        color='importance',
        color_continuous_scale=[(0, COLOR_NEG), (0.5, '#f0f0f0'), (1, COLOR_POS)],
        color_continuous_midpoint=0,
        custom_data=['Tooltip', 'importance']
    )

    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Affinity: %{customdata[0]}<extra></extra>',
        textinfo="label",
    )

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(t=30, l=10, r=10, b=10),
        coloraxis_showscale=False
    )
    return fig

# --- UI Components ---

def display_critic_profile(importances_df, selected_critic, critic_baseline):
    st.subheader(f"Critic Profile: {selected_critic}")
    
    # 1. Top Metrics
    c_base, c_count, c_top = st.columns(3)
    c_base.metric("Baseline Score", f"{critic_baseline:.2f}", help="The average score this critic gives to all games.")
    
    critic_data = importances_df[importances_df['critic_name'] == selected_critic].copy()
    
    if critic_data.empty:
        c_count.metric("Learned Tags", "0")
        c_top.metric("Top Driver", "N/A")
        st.warning("Not enough data to generate a profile. (User needs to rate more games).")
        return
    
    tag_df = critic_data[critic_data['feature'].str.startswith('tag__')].copy()
    tag_df['feature'] = tag_df['feature'].str.replace('tag__', '')
    
    c_count.metric("Learned Tags", len(tag_df))
    
    # SAFETY CHECK: Only calculate max if data exists
    if not tag_df.empty:
        top_tag = tag_df.loc[tag_df['importance'].abs().idxmax()]
        c_top.metric(f"Top Driver", top_tag['feature'], f"{top_tag['importance']:+.2f}")
    else:
        c_top.metric("Top Driver", "N/A")

    # 2. Treemap
    st.markdown("### Feature Affinity Map")
    st.caption("Visualizes the full spectrum of the critic's preferences. Size indicates magnitude of impact.")
    
    treemap_fig = plot_taste_treemap(tag_df, "")
    if treemap_fig:
        st.plotly_chart(treemap_fig, use_container_width=True)
    else:
        st.info("No tag data available for visualization.")

    with st.expander("View Raw Data"):
        if not tag_df.empty:
            display_df = tag_df[['feature', 'importance']].copy()
            display_df.columns = ['Tag', 'Affinity']
            st.dataframe(display_df.sort_values('Affinity', ascending=False), use_container_width=True)
        else:
            st.write("No data.")


def display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game, critic_baseline):
    st.subheader(f"Analysis: {selected_game}")
    
    record = merged_df[(merged_df['critic_name'] == selected_critic) & (merged_df['game_name'] == selected_game)]
    if record.empty:
        st.warning("No data found for this combination.")
        return
    record = record.iloc[0]
    
    predicted_score = record['predicted_score']
    
    # --- 1. Top Level Metrics (Comparison) ---
    with st.container(border=True):
        col_pred, col_act, col_delta = st.columns(3)
        
        # Predicted
        col_pred.metric("Predicted Score", f"{predicted_score:.2f}" if pd.notna(predicted_score) else "N/A")
        
        # Actual
        if record['upcoming']:
            act_val = "Upcoming"
            delta_val = None
        elif pd.notna(record['score']):
            act_val = f"{record['score']:.1f}"
            delta_val = record['score'] - predicted_score
        elif record['actual_skip']:
            act_val = "Skipped"
            delta_val = None
        else:
            act_val = "Pending"
            delta_val = None
            
        col_act.metric("Actual Outcome", act_val)
        
        # Delta
        if delta_val is not None:
            col_delta.metric("Model Error", f"{delta_val:+.2f}", help="Difference between Actual and Predicted.")
        else:
            col_delta.metric("Model Error", "N/A")

    # --- 2. Skip Probability Breakdown ---
    st.markdown("#### Skip Probability Analysis")
    skip_prob = record['predicted_skip_probability']
    
    if pd.notna(skip_prob):
        # Visual Progress Bar
        st.progress(skip_prob, text=f"Skip Probability: {skip_prob:.1%}")
        
        # Risk Factors
        game_meta = games_df[games_df['game_name'] == selected_game].iloc[0]
        game_features = get_game_features(game_meta)
        critic_profile = importances_df[importances_df['critic_name'] == selected_critic]
        
        contributions = []
        for f in game_features:
            match = critic_profile[critic_profile['feature'] == f]
            if not match.empty:
                contributions.append({'Feature': f.replace('tag__', '').replace('bin__', ''), 'Impact': match.iloc[0]['importance']})
        
        contrib_df = pd.DataFrame(contributions)
        
        if not contrib_df.empty:
            negatives = contrib_df[contrib_df['Impact'] < 0].sort_values('Impact', ascending=True)
            positives = contrib_df[contrib_df['Impact'] > 0].sort_values('Impact', ascending=False)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Risk Factors (Drivers for Skipping):**")
                if not negatives.empty:
                    for _, row in negatives.head(3).iterrows():
                        st.markdown(f"- {row['Feature']} ({row['Impact']:.2f})")
                else:
                    st.caption("No significant negative drivers found.")
            
            with c2:
                st.markdown("**Mitigating Factors (Drivers for Playing):**")
                if not positives.empty:
                    for _, row in positives.head(3).iterrows():
                        st.markdown(f"- {row['Feature']} (+{row['Impact']:.2f})")
                else:
                    st.caption("No significant positive drivers found.")
        else:
            st.caption("No strong profile matches found to assess risk.")
    else:
        st.info("Skip probability data unavailable.")

    st.divider()

    # --- 3. Feature Affinity Analysis ---
    st.markdown("#### Feature Affinity Analysis")
    st.caption("The chart below shows the critic's raw preference for this game's specific features.")
    st.info("Note: The Final Score is a weighted average of these preferences. Rare/Specific tags (e.g., 'Roguelike') are weighted more heavily than generic tags (e.g., 'Action').")

    if not contrib_df.empty:
        # Sort by absolute impact for visibility
        contrib_df['abs_impact'] = contrib_df['Impact'].abs()
        chart_df = contrib_df.sort_values('abs_impact', ascending=False).head(15)
        
        fig = plot_diverging_bar(chart_df, 'Impact', 'Feature', "")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for chart.")
    else:
        st.warning("No overlapping features found between the game and the critic's profile.")

# --- Main ---
def main():
    check_auth()
    session = get_sqla_session()
    
    st.title("Predictive Analytics")
    
    critic_names, games_df, merged_df, importances_df = load_prediction_data(session)
    game_names = games_df['game_name'].tolist()

    if merged_df.empty:
        st.error("No data available.")
        st.stop()

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_critic = st.selectbox("Select Critic", critic_names, index=0)
    
    # Calculate Baseline
    critic_rows = merged_df[merged_df['critic_name'] == selected_critic]
    actual_scores = critic_rows[critic_rows['score'].notna()]['score']
    
    # SAFETY: Default to 5.0 if no ratings exist
    critic_baseline = actual_scores.mean() if not actual_scores.empty else 5.0

    tab_profile, tab_game = st.tabs(["Critic Profile", "Game Analysis"])
    
    with tab_profile:
        display_critic_profile(importances_df, selected_critic, critic_baseline)
        
    with tab_game:
        selected_game = st.selectbox("Select Game", game_names, index=0)
        display_prediction_breakdown(merged_df, games_df, importances_df, selected_critic, selected_game, critic_baseline)
    
    st.write("")
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()