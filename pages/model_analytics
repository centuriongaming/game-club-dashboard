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
COLOR_GOOD = "#2ECC71"
COLOR_WARN = "#F1C40F"
COLOR_BAD = "#E74C3C"
COLOR_NEUTRAL = "#95A5A6"

# --- Data Loading ---
@st.cache_data
def load_analytics_data(_session):
    # Fetch all Predictions joined with Actual Ratings
    # We need: Critic, Game, Predicted Score, Predicted Skip Prob, Actual Score
    
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
        WHERE g.upcoming = FALSE -- Only analyze released games
    """)
    
    df = pd.read_sql(query, _session.bind)
    
    # Feature Engineering for Analytics
    # 1. actual_score is NaN implies a "Skip" (based on our DB logic)
    df['is_actual_skip'] = df['actual_score'].isna()
    
    # 2. Predicted Skip (Threshold > 50%)
    df['is_predicted_skip'] = df['predicted_skip_probability'] > 0.5
    
    # 3. Error Calculation (Only for games that were PLAYED)
    # Error = Predicted - Actual
    # Positive Error = Model was too Optimistic
    # Negative Error = Model was too Pessimistic
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
    bias = played_df['error'].mean() # Mean Signed Error
    
    # Correlation (how well do the ranks match?)
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
    Shows alignment.
    """
    played_df = df[df['is_actual_skip'] == False].copy()
    
    # Add jitter to avoid overplotting on integer scores
    played_df['actual_jitter'] = played_df['actual_score'] + np.random.normal(0, 0.1, size=len(played_df))
    
    fig = px.scatter(
        played_df,
        x='actual_jitter',
        y='predicted_score',
        color='error',
        color_continuous_scale='RdBu_r', # Red = High Error, Blue = Low Error
        hover_data=['game_name', 'critic_name', 'actual_score', 'predicted_score'],
        title="Actual vs. Predicted Scores"
    )
    
    # Perfect Prediction Line (y=x)
    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=10, y1=10,
        line=dict(color="Green", dash="dash"),
    )
    
    fig.update_layout(
        xaxis_title="Actual Score (with slight visual jitter)",
        yaxis_title="Predicted Score",
        coloraxis_colorbar=dict(title="Error"),
        height=500
    )
    return fig

def plot_error_histogram(df):
    """
    Shows the distribution of errors. 
    Ideally a bell curve centered at 0.
    """
    played_df = df[df['is_actual_skip'] == False]
    
    fig = px.histogram(
        played_df,
        x='error',
        nbins=20,
        title="Error Distribution (Residuals)",
        color_discrete_sequence=['#3498DB']
    )
    
    # Zero line
    fig.add_vline(x=0, line_width=3, line_color="black", annotation_text="Perfect Accuracy")
    
    fig.update_layout(
        xaxis_title="Prediction Error (Predicted - Actual)",
        yaxis_title="Count of Predictions",
        bargap=0.1
    )
    return fig

def plot_confusion_matrix(df):
    """
    Shows classification accuracy for Skips vs Plays.
    """
    # Create Confusion Categories
    conditions = [
        (df['is_actual_skip'] == True) & (df['is_predicted_skip'] == True),  # True Positive (Correct Skip)
        (df['is_actual_skip'] == False) & (df['is_predicted_skip'] == False), # True Negative (Correct Play)
        (df['is_actual_skip'] == True) & (df['is_predicted_skip'] == False), # False Negative (Missed Skip)
        (df['is_actual_skip'] == False) & (df['is_predicted_skip'] == True)  # False Positive (False Alarm)
    ]
    choices = ['Correctly Skipped', 'Correctly Played', 'Model said Play (User Skipped)', 'Model said Skip (User Played)']
    
    df['outcome'] = np.select(conditions, choices, default='Error')
    
    counts = df['outcome'].value_counts().reset_index()
    counts.columns = ['Outcome', 'Count']
    
    # Assign Colors
    color_map = {
        'Correctly Skipped': COLOR_GOOD,
        'Correctly Played': COLOR_GOOD,
        'Model said Play (User Skipped)': COLOR_BAD, # Annoying recommendation
        'Model said Skip (User Played)': COLOR_WARN  # Missed opportunity
    }
    
    fig = px.pie(
        counts, 
        values='Count', 
        names='Outcome',
        title="Skip Prediction Accuracy",
        color='Outcome',
        color_discrete_map=color_map,
        hole=0.4
    )
    return fig

# --- Main Layout ---

def main():
    check_auth()
    session = get_sqla_session()
    
    st.title("📊 Model Analytics")
    st.markdown("Transparency report: How accurately is the model predicting user behavior?")
    
    df = load_analytics_data(session)
    metrics = calculate_metrics(df)
    
    if metrics is None or metrics['N'] < 5:
        st.error("Not enough data to calculate analytics yet. Need at least 5 completed ratings.")
        st.stop()
        
    # --- 1. Global Metrics ---
    st.subheader("Global Performance")
    
    c1, c2, c3, c4 = st.columns(4)
    
    # MAE
    mae_val = metrics['MAE']
    mae_color = "normal"
    if mae_val < 1.0: mae_color = "off" # Greenish UI hint isn't available in standard metric but logic holds
    c1.metric(
        "Average Error (MAE)", 
        f"{mae_val:.2f} pts", 
        help="On average, how far off is the prediction? Lower is better.",
        delta="-0.1" if mae_val < 1.5 else "High Error",
        delta_color="inverse"
    )
    
    # Bias
    bias_val = metrics['Bias']
    bias_text = "Balanced"
    if bias_val > 0.5: bias_text = "Over-Optimistic"
    elif bias_val < -0.5: bias_text = "Pessimistic"
    
    c2.metric(
        "Model Bias", 
        f"{bias_val:+.2f}", 
        help="Positive = Model predicts too high. Negative = Model predicts too low.",
        delta=bias_text,
        delta_color="off"
    )
    
    # Classification Accuracy
    correct_preds = len(df[df['is_actual_skip'] == df['is_predicted_skip']])
    total_preds = len(df)
    acc = correct_preds / total_preds
    c3.metric(
        "Decision Accuracy", 
        f"{acc:.1%}", 
        help="How often did the model correctly guess if a user would Play or Skip?"
    )
    
    # Correlation
    c4.metric(
        "Correlation",
        f"{metrics['Correlation']:.2f}",
        help="1.0 = Perfect Rank Match. 0.0 = Random Guessing."
    )

    st.divider()

    # --- 2. Regression Analysis (Scores) ---
    st.subheader("🎯 Score Accuracy")
    col_scatter, col_hist = st.columns(2)
    
    with col_scatter:
        st.plotly_chart(plot_calibration_scatter(df), use_container_width=True)
        st.caption("**How to read:** Points on the dotted line are perfect predictions. Points **above** the line mean the model liked the game more than the user did.")
        
    with col_hist:
        st.plotly_chart(plot_error_histogram(df), use_container_width=True)
        st.caption("**How to read:** We want a tall spike at 0. If the curve is shifted left, the model is too harsh. If shifted right, it's too nice.")

    st.divider()

    # --- 3. Classification Analysis (Skips) ---
    st.subheader("🚫 Skip Prediction")
    col_conf, col_details = st.columns([1, 2])
    
    with col_conf:
        st.plotly_chart(plot_confusion_matrix(df), use_container_width=True)
        
    with col_details:
        st.markdown("#### Why this matters")
        st.write("""
        For a Game Club, **time is valuable**. 
        * **False Positives (Red in chart):** The model said "You'll like this!" but you skipped it. This wastes recommendation slots.
        * **False Negatives (Yellow in chart):** The model said "Skip it," but you played it anyway. This means the model might be hiding gems.
        """)
        
        # Show stats table
        confusion = pd.crosstab(df['is_actual_skip'], df['is_predicted_skip'], rownames=['Actual Skip'], colnames=['Predicted Skip'])
        st.dataframe(confusion, use_container_width=True)

    st.divider()

    # --- 4. Leaderboard (Who is hardest to predict?) ---
    st.subheader("🧩 Difficulty by Critic")
    st.caption("Which users are the most unpredictable?")
    
    critic_stats = []
    for critic, group in df.groupby('critic_name'):
        played_group = group[group['is_actual_skip'] == False]
        if len(played_group) < 3: continue # Skip users with little data
        
        c_mae = played_group['abs_error'].mean()
        c_bias = played_group['error'].mean()
        
        critic_stats.append({
            "Critic": critic,
            "Avg Error (Points)": c_mae,
            "Bias": c_bias,
            "Tendency": "Hard to Please" if c_bias > 0.5 else ("Easy to Please" if c_bias < -0.5 else "Neutral"),
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

    # Footer
    st.write("")
    if st.button("Log out"):
        st.session_state["password_correct"] = False
        st.rerun()

if __name__ == "__main__":
    main()