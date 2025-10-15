# pages/predictions.py
import streamlit as st
import pandas as pd
import sqlalchemy as sa
import plotly.express as px
from utils import check_auth, get_sqla_session
from database_models import Critic, CriticPrediction, CriticFeatureImportance

# --- Page & Data Configuration ---
st.set_page_config(page_title="Predictions", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_prediction_data(_session):
    """Loads all data needed for the predictions page from the database."""
    # Query for a list of all critics
    critics_df = pd.read_sql(sa.select(Critic.critic_name).order_by(Critic.critic_name), _session.bind)
    critic_names = critics_df['critic_name'].tolist()

    # Query for all game predictions, joining with critics to get their names
    predictions_stmt = (
        sa.select(
            CriticPrediction.name.label("game_name"),
            Critic.critic_name,
            CriticPrediction.predicted_score,
            CriticPrediction.predicted_skip_probability
        )
        .join(Critic, Critic.id == CriticPrediction.critic_id)
    )
    predictions_df = pd.read_sql(predictions_stmt, _session.bind)

    # Query for all feature importances for the prediction models
    importances_stmt = (
        sa.select(
            CriticFeatureImportance.feature,
            CriticFeatureImportance.importance,
            Critic.critic_name,
            CriticFeatureImportance.model_type
        )
        .join(Critic, Critic.id == CriticFeatureImportance.critic_id)
    )
    importances_df = pd.read_sql(importances_stmt, _session.bind)

    return critic_names, predictions_df, importances_df

# --- UI Component Functions ---

def display_prediction_tables(predictions_df, selected_critic):
    """
    Displays the predicted scores and skip probabilities for the selected critic
    in two side-by-side tables.
    """
    st.subheader(f"Game Predictions for {selected_critic}")
    
    # Filter the main dataframe for the chosen critic
    critic_preds = predictions_df[predictions_df['critic_name'] == selected_critic].copy()

    if critic_preds.empty:
        st.info("No predictions available for this critic.")
        return

    # Convert skip probability from a 0-1 scale to a 0-100 scale for the progress bar
    critic_preds['predicted_skip_probability'] *= 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Predicted Scores")
        st.caption("The model's prediction for the score this critic would give.")
        # Prepare and display the scores dataframe
        score_df = critic_preds[['game_name', 'predicted_score']].sort_values('predicted_score', ascending=False).dropna()
        st.dataframe(
            score_df,
            column_config={
                "game_name": "Game",
                "predicted_score": st.column_config.ProgressColumn("Predicted Score", format="%.2f", min_value=0, max_value=10)
            },
            hide_index=True, use_container_width=True
        )

    with col2:
        st.markdown("##### Skip Likelihood")
        st.caption("The predicted probability that the critic will not rate a game.")
        # Prepare and display the skip probability dataframe
        skip_df = critic_preds[['game_name', 'predicted_skip_probability']].sort_values('predicted_skip_probability', ascending=False).dropna()
        st.dataframe(
            skip_df,
            column_config={
                "game_name": "Game",
                "predicted_skip_probability": st.column_config.ProgressColumn("Skip Probability", format="%.1f%%", min_value=0, max_value=100)
            },
            hide_index=True, use_container_width=True
        )

def display_feature_importance_charts(importances_df, selected_critic):
    """
    Displays feature importance bar charts for the selected critic's models.
    It creates a separate tab for each model type found (e.g., score vs. skip).
    """
    st.subheader(f"Model Insights for {selected_critic}")
    st.caption("These charts show the most influential factors in the prediction models for this critic.")
    
    critic_importances = importances_df[importances_df['critic_name'] == selected_critic]

    if critic_importances.empty:
        st.info("No feature importance data is available for this critic.")
        return

    # Create tabs for each unique model type (e.g., "Score Prediction", "Skip Prediction")
    model_types = critic_importances['model_type'].unique()
    tab_list = st.tabs([f"{m.replace('_', ' ').title()} Model" for m in model_types])

    for i, model_type in enumerate(model_types):
        with tab_list[i]:
            # Filter for the specific model type, sort by importance, and take the top 15 features
            model_df = critic_importances[critic_importances['model_type'] == model_type].sort_values('importance', ascending=False).head(15)
            
            # Create a horizontal bar chart
            fig = px.bar(
                model_df,
                x='importance',
                y='feature',
                orientation='h',
                title=f"Top Features for {model_type.replace('_', ' ').title()} Prediction"
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'}, # Ensure the most important feature is at the top
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Main Page ---
def main():
    """Renders the main predictions page."""
    check_auth()
    session = get_sqla_session()
    
    st.title("🤖 Predictive Analytics")

    # Load all necessary data at once
    critic_names, predictions_df, importances_df = load_prediction_data(session)

    # Stop if no data is available to display
    if not critic_names:
        st.warning("No critic data found. Predictions cannot be displayed.")
        st.stop()
        
    # --- User Controls ---
    st.sidebar.header("Select a Critic")
    selected_critic = st.sidebar.selectbox(
        "Choose a critic to see their personalized predictions:",
        critic_names,
        index=0,
        label_visibility="collapsed"
    )

    # --- Page Content ---
    if selected_critic:
        with st.container(border=True):
            display_prediction_tables(predictions_df, selected_critic)
        
        st.markdown("---")

        with st.container(border=True):
            display_feature_importance_charts(importances_df, selected_critic)

    # --- Logout Button ---
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()