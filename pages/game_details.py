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
st.write("Select a game to see its detailed rating breakdown.")

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
    
    # KPIs for the selected game
    game_metrics_query = f"""
        SELECT
            AVG(score) as average_score,
            COUNT(score) as ratings_given,
            (SELECT COUNT(*) FROM critics) as total_critics
        FROM ratings WHERE game_id = {selected_game_id};
    """
    game_metrics_df = conn.query(game_metrics_query)
    avg_score = game_metrics_df['average_score'][0]
    play_rate = (game_metrics_df['ratings_given'][0] / game_metrics_df['total_critics'][0]) * 100
    
    st.header(f"Metrics for {selected_game_name}")
    col1, col2 = st.columns(2)
    col1.metric("Average Score", f"{avg_score:.2f}")
    col2.metric("Play Rate", f"{play_rate:.1f}%")

    # Individual scores for the selected game
    st.header("Individual Critic Scores")
    critic_scores_query = f"""
        SELECT c.critic_name, r.score
        FROM critics c
        LEFT JOIN ratings r ON c.id = r.critic_id AND r.game_id = {selected_game_id}
        ORDER BY c.critic_name;
    """
    critic_scores_df = conn.query(critic_scores_query)
    st.dataframe(critic_scores_df, hide_index=True, use_container_width=True)

st.divider()

# --- Game Controversy Analysis ---
st.header("Game Controversy Analysis")
st.write("Controversy is measured by the standard deviation of scores. A high value means critics disagreed widely.")
controversy_query = """
    SELECT 
        g.game_name,
        STDDEV(r.score) as controversy_score,
        COUNT(r.score) as number_of_ratings
    FROM ratings r
    JOIN games g ON r.game_id = g.id
    GROUP BY g.game_name
    HAVING COUNT(r.score) > 1
    ORDER BY controversy_score DESC;
"""
controversy_df = conn.query(controversy_query)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Most Controversial Games")
    st.dataframe(controversy_df.head(5), hide_index=True)
with col2:
    st.subheader("Least Controversial Games")
    st.dataframe(controversy_df.sort_values("controversy_score", ascending=True).head(5), hide_index=True)