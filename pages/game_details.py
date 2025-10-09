# pages/game_details.py
import streamlit as st
import pandas as pd

# --- Authentication Check ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Game Details", layout="wide")
st.title("Game Details")
st.write("Select a game to see its detailed rating breakdown and individual critic scores.")

# --- Database Connection ---
try:
    conn = st.connection("mydb", type="sql")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# --- Game Selection ---
games_list_df = conn.query("SELECT id, game_name FROM games WHERE upcoming IS FALSE ORDER BY game_name;")
game_dict = pd.Series(games_list_df.id.values, index=games_list_df.game_name).to_dict()

selected_game_name = st.selectbox("Select a Game:", games_list_df['game_name'])

if selected_game_name:
    selected_game_id = game_dict[selected_game_name]
    
    # --- Fetch Detailed Game Metrics ---
    game_metrics_query = f"""
        SELECT
            AVG(score) as average_score,
            STDDEV(score) as controversy_score,
            COUNT(score) as ratings_given,
            (SELECT COUNT(*) FROM critics) as total_critics
        FROM ratings
        WHERE game_id = {selected_game_id};
    """
    game_metrics_df = conn.query(game_metrics_query)
    
    avg_score = game_metrics_df['average_score'][0]
    controversy = game_metrics_df['controversy_score'][0]
    play_rate = (game_metrics_df['ratings_given'][0] / game_metrics_df['total_critics'][0]) * 100
    
    st.header(f"Metrics for {selected_game_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Score", f"{avg_score:.2f}" if avg_score is not None else "N/A")
    col2.metric("Play Rate", f"{play_rate:.0f}%")
    col3.metric("Controversy (Std Dev)", f"{controversy:.3f}" if controversy is not None else "N/A")

    # --- Display Individual Critic Scores ---
    st.header("Individual Critic Scores")
    critic_scores_query = f"""
        SELECT 
            c.critic_name,
            r.score
        FROM critics c
        LEFT JOIN ratings r ON c.id = r.critic_id AND r.game_id = {selected_game_id}
        ORDER BY c.critic_name;
    """
    critic_scores_df = conn.query(critic_scores_query)

    st.dataframe(
        critic_scores_df,
        column_config={"score": "Score", "critic_name": "Critic"},
        hide_index=True, use_container_width=True
    )