# pages/predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy as sa
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating, CriticPrediction, CriticFeatureImportance

# --- Page & Data Configuration ---
st.set_page_config(page_title="Predictions", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_prediction_data(_session):
    """Loads and merges all data needed for the predictions page."""
    # Base queries
    critics_df = pd.read_sql(sa.select(Critic.id, Critic.critic_name).order_by(Critic.critic_name), _session.bind)
    games_df = pd.read_sql(sa.select(Game.id, Game.game_name, Game.upcoming).order_by(Game.game_name), _session.bind)
    predictions_df = pd.read_sql(
        sa.select(CriticPrediction.name.label("game_name"), Critic.critic_name, CriticPrediction.predicted_score, CriticPrediction.predicted_skip_probability)
        .join(Critic, Critic.id == CriticPrediction.critic_id), 
        _session.bind
    )
    ratings_df = pd.read_sql(
        sa.select(Critic.critic_name, Game.game_name, Rating.score)
        .join(Critic, Critic.id == Rating.critic_id)
        .join(Game, Game.id == Rating.game_id),
        _session.bind
    )
    importances_df = pd.read_sql(
        sa.select(CriticFeatureImportance.feature, CriticFeatureImportance.importance, Critic.critic_name, CriticFeatureImportance.model_type)
        .join(Critic, Critic.id == CriticFeatureImportance.critic_id),
        _session.bind
    )

    # Create a scaffold of all possible critic-game combinations to ensure data integrity
    scaffold_df = pd.MultiIndex.from_product(
        [critics_df['critic_name'], games_df['game_name']],
        names=['critic_name', 'game_name']
    ).to_frame(index=False)

    # Join the scaffold with all other data sources
    merged_df = pd.merge(scaffold_df, games_df[['game_name', 'upcoming']], on='game_name', how='left')
    merged_df = pd.merge(merged_df, predictions_df, on=['critic_name', 'game_name'], how='left')
    merged_df = pd.merge(merged_df, ratings_df, on=['critic_name', 'game_name'], how='left')
    
    # Engineer features for analysis
    merged_df['actual_skip'] = merged_df['score'].isna()
    
    return critics_df['critic_name'].tolist(), games_df['game_name'].tolist(), merged_df, importances_df

# --- UI Component Functions ---

def display_single_prediction(df, selected_critic, selected_game):
    """Displays the predicted vs. actual outcomes for a single critic/game pair."""
    st.subheader(f"Prediction Details")
    
    record = df[(df['critic_name'] == selected_critic) & (df['game_name'] == selected_game)]
    
    if record.empty:
        st.warning("No prediction is available for this combination.")
        return
        
    record = record.iloc[0]

    # Handle cases where a prediction could not be generated
    if pd.isna(record['predicted_score']):
        st.info("This user hasn't rated enough games yet for predictions.")
        return

    with st.container(border=True):
        col1, col2 = st.columns(2)
        is_upcoming = record['upcoming']
        
        # --- Score Prediction Column ---
        with col1:
            pred_score = record['predicted_score']
            st.metric(label="Predicted Score", value=f"{pred_score:.2f}")
            
            # Only show actuals for games that are not upcoming
            if not is_upcoming:
                actual_score = record['score']
                if pd.notna(actual_score):
                    delta = actual_score - pred_score
                    st.metric(label="Actual Score", value=f"{actual_score:.2f}", delta=f"{delta:.2f} (Actual - Predicted)")
                else:
                    st.metric(label="Actual Score", value="N/A (Skipped)")

        # --- Skip Prediction Column ---
        with col2:
            pred_prob = record['predicted_skip_probability']
            st.metric(label="Predicted Skip Likelihood", value=f"{pred_prob*100:.1f}%")

            # Only show actuals for games that are not upcoming
            if not is_upcoming:
                if record['actual_skip']:
                    st.metric(label="Actual Behavior", value="Skipped")
                else:
                    st.metric(label="Actual Behavior", value="Did Not Skip")
        
        if is_upcoming:
            st.caption("Actual results are not displayed for upcoming games.")

def display_model_performance_stats(df):
    """Calculates and displays overall model performance metrics with simplified explanations."""
    st.subheader("Overall Model Performance")

    with st.container(border=True):
        tab1, tab2 = st.tabs(["Score Prediction Model", "Skip Prediction Model"])

        # --- Score Model Stats ---
        with tab1:
            # Drop rows where actual or predicted score is missing to compare apples to apples
            rated_games_df = df.dropna(subset=['score', 'predicted_score'])
            if not rated_games_df.empty:
                mae = mean_absolute_error(rated_games_df['score'], rated_games_df['predicted_score'])
                rmse = np.sqrt(mean_squared_error(rated_games_df['score'], rated_games_df['predicted_score']))
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric(
                    "Average Score Error", 
                    f"{mae:.3f}", 
                    help="On average, the model's score predictions are off by this many points. A smaller number means the model is more accurate."
                )
                m_col2.metric(
                    "Large Error Penalty", 
                    f"{rmse:.3f}", 
                    help="This also measures error, but it gives a much bigger penalty for predictions that were wildly wrong. A smaller number is better."
                )
            else:
                st.info("Not enough actual scores available to calculate performance.")
        
        # --- Skip Model Stats ---
        with tab2:
            # Filter out upcoming games for a fair assessment of skip accuracy
            past_games_df = df[df['upcoming'] == False].copy()
            
            y_true = past_games_df['actual_skip']
            
            # --- FIX: Fill missing predictions with a neutral 0.5 value instead of dropping them ---
            y_prob = past_games_df['predicted_skip_probability'].fillna(0.5).clip(1e-10, 1-1e-10)
            
            loss = log_loss(y_true, y_prob)
            accuracy = ((y_prob > 0.5) == y_true).mean()
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric(
                "Prediction Confidence Score", 
                f"{loss:.3f}", 
                help="This measures how 'confident' the skip prediction model is. It heavily penalizes the model for being very confident about a wrong prediction. A smaller number means the model is better calibrated."
            )
            m_col2.metric(
                "Overall 'Skip vs. Rate' Accuracy", 
                f"{accuracy:.2%}", 
                help="The percentage of times the model correctly predicted whether a critic would skip rating a game or not."
            )
            )
def display_feature_importance_charts(importances_df, selected_critic):
    """Displays feature importance bar charts for the selected critic's models."""
    st.subheader(f"Model Insights for {selected_critic}")
    st.caption("These charts show the most influential factors in the prediction models for this critic.")

    with st.container(border=True):
        critic_importances = importances_df[importances_df['critic_name'] == selected_critic]

        if critic_importances.empty:
            st.info("No feature importance data is available for this critic.")
            return

        model_types = critic_importances['model_type'].unique()
        tab_list = st.tabs([f"{m.replace('_', ' ').title()} Model" for m in model_types])

        for i, model_type in enumerate(model_types):
            with tab_list[i]:
                model_df = critic_importances[critic_importances['model_type'] == model_type].sort_values('importance', ascending=False).head(15)
                fig = px.bar(model_df, x='importance', y='feature', orientation='h', title=f"Top Features for {model_type.replace('_', ' ').title()}")
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Importance Score", yaxis_title=None, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

# --- Main Page ---
def main():
    """Renders the main predictions page."""
    check_auth()
    session = get_sqla_session()
    
    st.title("Predictive Analytics")

    critic_names, game_names, merged_df, importances_df = load_prediction_data(session)

    if merged_df.empty:
        st.warning("No prediction data found. Page cannot be displayed.")
        st.stop()
        
    # --- User Controls ---
    st.markdown("### Explore a Single Prediction")
    col1, col2 = st.columns(2)
    with col1:
        selected_critic = st.selectbox("Select a Critic", critic_names, index=0)
    with col2:
        selected_game = st.selectbox("Select a Game", game_names, index=0)
    
    st.divider()

    # --- Page Content ---
    if selected_critic and selected_game:
        display_single_prediction(merged_df, selected_critic, selected_game)
    
    st.divider()

    display_model_performance_stats(merged_df)

    st.divider()

    if selected_critic:
        display_feature_importance_charts(importances_df, selected_critic)

    # --- Logout Button ---
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()