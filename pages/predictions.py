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
    """Loads and merges all data needed for the predictions page using IDs for robust joins."""
    # Base queries - Fetching with IDs
    critics_df = pd.read_sql(sa.select(Critic.id.label('critic_id'), Critic.critic_name).order_by(Critic.critic_name), _session.bind)
    games_df = pd.read_sql(sa.select(Game.id.label('game_id'), Game.game_name, Game.upcoming).order_by(Game.game_name), _session.bind)
    predictions_df = pd.read_sql(sa.select(CriticPrediction.critic_id, CriticPrediction.id.label('game_id'), CriticPrediction.predicted_score, CriticPrediction.predicted_skip_probability), _session.bind)
    ratings_df = pd.read_sql(sa.select(Rating.critic_id, Rating.game_id, Rating.score), _session.bind)
    importances_df = pd.read_sql(
        sa.select(CriticFeatureImportance.feature, CriticFeatureImportance.importance, Critic.critic_name, CriticFeatureImportance.model_type)
        .join(Critic, Critic.id == CriticFeatureImportance.critic_id),
        _session.bind
    )

    # Create a scaffold of all possible critic_id-game_id combinations
    scaffold_df = pd.MultiIndex.from_product(
        [critics_df['critic_id'], games_df['game_id']],
        names=['critic_id', 'game_id']
    ).to_frame(index=False)

    # Merge all data onto the scaffold using IDs
    merged_df = pd.merge(scaffold_df, critics_df, on='critic_id', how='left')
    merged_df = pd.merge(merged_df, games_df, on='game_id', how='left')
    merged_df = pd.merge(merged_df, predictions_df, on=['critic_id', 'game_id'], how='left')
    merged_df = pd.merge(merged_df, ratings_df, on=['critic_id', 'game_id'], how='left')
    
    # Engineer features for analysis
    merged_df['actual_skip'] = merged_df['score'].isna()
    
    return critics_df['critic_name'].tolist(), games_df['game_name'].tolist(), merged_df, importances_df

# --- UI Component Functions ---

def display_single_prediction(df, selected_critic, selected_game):
    """Displays the predicted vs. actual outcomes for a single critic/game pair."""
    record = df[(df['critic_name'] == selected_critic) & (df['game_name'] == selected_game)]
    
    if record.empty:
        st.warning("No prediction is available for this combination.")
        return
        
    record = record.iloc[0]

    # Differentiate between missing predictions for past vs. upcoming games
    if pd.isna(record['predicted_score']):
        if not record['upcoming']:
            st.info("This user hasn't rated enough games yet for predictions.")
        else:
            st.info("A prediction for this upcoming game is not yet available.")
        return

    with st.container(border=True):
        col1, col2 = st.columns(2)
        is_upcoming = record['upcoming']
        
        with col1:
            pred_score = record['predicted_score']
            st.metric(label="Predicted Score", value=f"{pred_score:.2f}")
            
            if not is_upcoming:
                actual_score = record['score']
                if pd.notna(actual_score):
                    delta = actual_score - pred_score
                    st.metric(label="Actual Score", value=f"{actual_score:.2f}", delta=f"{delta:.2f} (Actual - Predicted)")
                else:
                    st.metric(label="Actual Score", value="N/A (Skipped)")

        with col2:
            pred_prob = record['predicted_skip_probability']
            st.metric(label="Predicted Skip Likelihood", value=f"{pred_prob*100:.1f}%")

            if not is_upcoming:
                if record['actual_skip']:
                    st.metric(label="Actual Behavior", value="Skipped")
                else:
                    st.metric(label="Actual Behavior", value="Did Not Skip")
        
        if is_upcoming:
            st.caption("Actual results are not displayed for upcoming games.")

def display_model_performance_stats(df, selected_critic):
    """Calculates and displays model performance metrics for the selected critic."""
    st.subheader(f"Model Performance for {selected_critic}")

    with st.container(border=True):
        tab1, tab2 = st.tabs(["Score Prediction Model", "Skip Prediction Model"])

        # Filter data for the selected critic
        critic_df = df[df['critic_name'] == selected_critic].copy()

        with tab1:
            rated_games_df = critic_df.dropna(subset=['score', 'predicted_score'])
            if not rated_games_df.empty:
                mae = mean_absolute_error(rated_games_df['score'], rated_games_df['predicted_score'])
                rmse = np.sqrt(mean_squared_error(rated_games_df['score'], rated_games_df['predicted_score']))
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Average Score Error", f"{mae:.3f}", help="On average, the model's score predictions are off by this many points.")
                m_col2.metric("Large Error Penalty", f"{rmse:.3f}", help="This also measures error, but it gives a bigger penalty for wildly wrong predictions.")

                with st.expander("Show Data and Calculation"):
                    debug_df = rated_games_df[['game_name', 'score', 'predicted_score']].copy()
                    debug_df['absolute_error'] = (debug_df['predicted_score'] - debug_df['score']).abs()
                    
                    st.markdown(f"**Calculation:** $$ \\frac{{\\sum |\\text{{actual}} - \\text{{predicted}} |}}{{\\text{{count}}}} = \\frac{{{debug_df['absolute_error'].sum():.2f}}}{{{len(debug_df)}}} = {mae:.3f} $$")
                    st.dataframe(debug_df, use_container_width=True)

            else:
                st.info(f"Not enough rated games with predictions for {selected_critic} to calculate score model performance.")
        
        with tab2:
            past_games_df = critic_df[critic_df['upcoming'] == False].copy()
            past_games_df.dropna(subset=['predicted_skip_probability'], inplace=True)
            
            if not past_games_df.empty:
                y_true = past_games_df['actual_skip']
                y_prob = past_games_df['predicted_skip_probability'].clip(1e-10, 1-1e-10) 
                
                loss = log_loss(y_true, y_prob)
                accuracy = ((y_prob > 0.5) == y_true).mean()
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Prediction Confidence Score", f"{loss:.3f}", help="Measures how 'confident' the model is. Penalizes being very confident about a wrong prediction.")
                m_col2.metric("'Skip vs. Rate' Accuracy", f"{accuracy:.2%}", help="The percentage of times the model correctly predicted whether a critic would skip a game.")

                with st.expander("Show Data and Calculation"):
                    debug_df = past_games_df[['game_name', 'score', 'predicted_skip_probability', 'actual_skip']].copy()
                    debug_df['model_prediction_is_skip'] = debug_df['predicted_skip_probability'] > 0.5
                    debug_df['prediction_is_correct'] = debug_df['model_prediction_is_skip'] == debug_df['actual_skip']

                    st.markdown(f"**Calculation:** $$ \\frac{{\\text{{Correct Predictions}}}}{{\\text{{Total Predictions}}}} = \\frac{{{debug_df['prediction_is_correct'].sum()}}}{{{len(debug_df)}}} = {accuracy:.2%} $$")
                    st.dataframe(debug_df, use_container_width=True)
            else:
                st.info(f"Not enough past games with predictions for {selected_critic} to calculate skip model performance.")

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
    
    # --- Critic Selection ---
    st.markdown("### Select Critic")
    selected_critic = st.selectbox("Select a Critic", critic_names, index=0, label_visibility="collapsed")
    st.divider()

    # --- Overall and Critic-Specific Stats ---
    if selected_critic:
        display_model_performance_stats(merged_df, selected_critic)
        st.divider()
        display_feature_importance_charts(importances_df, selected_critic)
        st.divider()

    # --- Single Game Prediction ---
    st.markdown("### Explore a Single Prediction")
    st.caption(f"Showing predictions for **{selected_critic}**.")
    selected_game = st.selectbox("Select a Game", game_names, index=0)
    
    if selected_critic and selected_game:
        display_single_prediction(merged_df, selected_critic, selected_game)

    # --- Logout Button ---
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()