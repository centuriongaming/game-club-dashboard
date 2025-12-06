import streamlit as st
import pandas as pd
import numpy as np
import sqlalchemy as sa
import plotly.express as px
import plotly.graph_objects as go
from utils import check_auth, get_sqla_session
from database_models import Critic, Game, Rating, CriticPrediction

# --- Page Config ---
st.set_page_config(page_title="Model Analytics", layout="wide")

# --- Constants ---
# Professional color palette
COLOR_CORRECT = "#2ECC71" # Green
COLOR_ERROR = "#E74C3C"   # Red

# --- Data Loading ---
@st.cache_data
def load_analytics_data(_session):
    query = sa.text("""
        SELECT 
            c.critic_name,
            g.game_name,
            cp.predicted_score,
            cp.predicted_skip_probability,
            r.score as actual_score
        FROM critic_predictions cp
        JOIN critics c ON cp.critic_id = c.id
        JOIN games g ON cp.id = g.id
        LEFT JOIN ratings r ON cp.critic_id = r.critic_id AND cp.id = r.game_id
        WHERE g.upcoming = FALSE 
    """)
    
    df = pd.read_sql(query, _session.bind)
    
    # Logic: If actual_score is NaN, it counts as a Skip
    df['is_actual_skip'] = df['actual_score'].isna()
    df['is_predicted_skip'] = df['predicted_skip_probability'] > 0.5
    
    # Calculate Error (Predicted - Actual) for played games only
    df['error'] = df['predicted_score'] - df['actual_score']
    df['abs_error'] = df['error'].abs()
    
    return df

# --- Metric Helpers ---
def calculate_metrics(df):
    played_df = df[df['is_actual_skip'] == False]
    
    if played_df.empty:
        return None

    mae = played_df['abs_error'].mean()
    rmse = np.sqrt((played_df['error'] ** 2).mean())
    bias = played_df['error'].mean()
    correlation = played_df['predicted_score'].corr(played_df['actual_score'])
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "Correlation": correlation,
        "N": len(played_df)
    }

# --- Visualizations ---

def plot_calibration_scatter(df):
    """
    Actual vs Predicted Scatter Plot.
    """
    played_df = df[df['is_actual_skip'] == False].copy()
    
    # Jitter for visualization
    played_df['actual_jitter'] = played_df['actual_score'] + np.random.normal(0, 0.1, size=len(played_df))
    
    fig = px.scatter(
        played_df,
        x='actual_jitter',
        y='predicted_score',
        color='error',
        color_continuous_scale='RdBu_r', 
        hover_data=['game_name', 'critic_name', 'actual_score', 'predicted_score'],
        title="Actual vs. Predicted Scores"
    )
    
    # Perfect Prediction Line
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=10, y1=10,
        line=dict(color="Green", dash="dash"),
    )
    
    fig.update_layout(
        xaxis_title="Actual Score",
        yaxis_title="Predicted Score",
        coloraxis_colorbar=dict(title="Error"),
        height=500
    )
    return fig

def plot_error_histogram(df):
    """
    Distribution of errors.
    """
    played_df = df[df['is_actual_skip'] == False]
    
    fig = px.histogram(
        played_df,
        x='error',
        nbins=20,
        title="Error Distribution (Residuals)",
        color_discrete_sequence=['#3498DB']
    )
    
    fig.add_vline(x=0, line_width=2, line_color="black")
    
    fig.update_layout(
        xaxis_title="Prediction Error",
        yaxis_title="Count",
        bargap=0.1
    )
    return fig

def plot_confusion_heatmap(df):
    """
    Standard Confusion Matrix Heatmap.
    Visualizes True Positives, False Positives, etc.
    """
    # Create the matrix
    # Rows: Actual, Cols: Predicted
    confusion = pd.crosstab(
        df['is_actual_skip'].replace({True: 'Skipped', False: 'Played'}), 
        df['is_predicted_skip'].replace({True: 'Skipped', False: 'Played'}), 
        rownames=['Actual'], 
        colnames=['Predicted']
    )

    # Ensure all columns/rows exist even if data is missing
    for label in ['Played', 'Skipped']:
        if label not in confusion.columns: confusion[label] = 0
        if label not in confusion.index: confusion.loc[label] = 0
        
    # Reorder for logical flow: Played -> Skipped
    confusion = confusion.reindex(index=['Played', 'Skipped'], columns=['Played', 'Skipped'])
    
    # Convert to matrix for heatmap
    z = confusion.values
    x = confusion.columns.tolist()
    y = confusion.index.tolist()

    # Create annotation text (the counts)
    z_text = [[str(y) for y in x] for x in z]

    fig = px.imshow(
        z, 
        x=x, 
        y=y, 
        color_continuous_scale='Blues',
        text_auto=True,
        title="Skip Prediction Accuracy (Confusion Matrix)"
    )
    
    fig.update_layout(
        xaxis_title="Model Predicted",
        yaxis_title="User Actually",
        height=400
    )
    return fig

# --- Main Layout ---

def main():
    check_auth()
    session = get_sqla_session()
    
    st.title("Model Analytics")
    st.markdown("Transparency report regarding model performance and accuracy.")
    
    df = load_analytics_data(session)
    metrics = calculate_metrics(df)
    
    if metrics is None or metrics['N'] < 5:
        st.error("Insufficient data. Need at least 5 completed ratings.")
        st.stop()
        
    # --- 1. Global Metrics ---
    st.subheader("Global Performance")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # MAE
    mae_val = metrics['MAE']
    c1.metric(
        "Average Error (MAE)", 
        f"{mae_val:.2f} pts", 
        help="On average, how far off is the prediction?"
    )
    
    # Bias
    bias_val = metrics['Bias']
    bias_text = "Balanced"
    if bias_val > 0.5: bias_text = "Optimistic"
    elif bias_val < -0.5: bias_text = "Pessimistic"
    
    c2.metric(
        "Model Bias", 
        f"{bias_val:+.2f}", 
        help="Positive means model over-predicts. Negative means under-predicts.",
        delta=bias_text,
        delta_color="off"
    )
    
    # Accuracy
    correct_preds = len(df[df['is_actual_skip'] == df['is_predicted_skip']])
    total_preds = len(df)
    acc = correct_preds / total_preds
    c3.metric(
        "Decision Accuracy", 
        f"{acc:.1%}", 
        help="Percentage of correct Play vs. Skip predictions."
    )
    
    # Correlation
    c4.metric(
        "Correlation",
        f"{metrics['Correlation']:.2f}",
        help="Rank correlation between predicted and actual scores."
    )

    st.divider()

    # --- 2. Regression Analysis (Scores) ---
    st.subheader("Score Accuracy")
    col_scatter, col_hist = st.columns(2)
    
    with col_scatter:
        st.plotly_chart(plot_calibration_scatter(df), use_container_width=True)
        st.caption("Points on the dotted line are perfect predictions.")
        
    with col_hist:
        st.plotly_chart(plot_error_histogram(df), use_container_width=True)
        st.caption("A spike at 0 indicates high accuracy.")

    st.divider()

    # --- 3. Classification Analysis (Skips) ---
    st.subheader("Skip Prediction")
    col_conf, col_details = st.columns([1, 2])
    
    with col_conf:
        st.plotly_chart(plot_confusion_heatmap(df), use_container_width=True)
        
    with col_details:
        st.markdown("#### Performance Breakdown")
        st.write("""
        This heatmap shows where the model succeeds and fails in predicting whether a user will play a game.
        
        * **Diagonal (Dark Blue):** Correct predictions.
        * **Off-Diagonal (Light Blue):** Errors.
        
        False Positives (Model said 'Played', User 'Skipped') are generally more annoying to users than False Negatives, as they clutter recommendations.
        """)

    st.divider()

    # --- 4. Leaderboard ---
    st.subheader("Difficulty by Critic")
    st.caption("Identifies which users deviate most often from the model's predictions.")
    
    critic_stats = []
    for critic, group in df.groupby('critic_name'):
        played_group = group[group['is_actual_skip'] == False]
        if len(played_group) < 3: continue 
        
        c_mae = played_group['abs_error'].mean()
        c_bias = played_group['error'].mean()
        
        critic_stats.append({
            "Critic": critic,
            "Avg Error (Points)": c_mae,
            "Bias": c_bias,
            "Rated Games": len(played_group)
        })
        
    critic_df = pd.DataFrame(critic_stats).sort_values("Avg Error (Points)", ascending=False)
    
    st.dataframe(
        critic_df,
        column_config={
            "Avg Error (Points)": st.column_config.NumberColumn(format="%.2f"),
            "Bias": st.column_config.NumberColumn(format="%+.2f"),
        },
        use_container_width=True,
        hide_index=True
    )

    st.write("")
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()