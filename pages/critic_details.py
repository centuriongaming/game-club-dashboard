# pages/critic_details.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Authentication Check ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Critic Details", layout="wide")
st.title("Critic Details & Analysis")

# --- Database Connection ---
try:
    conn = st.connection("mydb", type="sql")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# --- Critic Participation Table ---
st.header("Critics by Participation")
critic_participation_query = """
    SELECT
        c.critic_name,
        COUNT(r.score) AS ratings_given,
        (SELECT COUNT(*) FROM games WHERE upcoming IS FALSE) AS total_games,
        AVG(r.score) AS average_score
    FROM critics c
    LEFT JOIN ratings r ON c.id = r.critic_id
    GROUP BY c.critic_name
    ORDER BY ratings_given DESC;
"""
critic_participation_df = conn.query(critic_participation_query)
critic_participation_df['participation_rate'] = (critic_participation_df['ratings_given'] / critic_participation_df['total_games']) * 100
st.dataframe(
    critic_participation_df,
    column_config={"average_score": st.column_config.ProgressColumn("Average Score",format="%.2f",min_value=0,max_value=10),
                   "participation_rate": st.column_config.ProgressColumn("Participation Rate",format="%.1f%%",min_value=0,max_value=100)},
    hide_index=True
)

# --- Visual Breakdowns ---
st.header("Visual Breakdowns")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Nominations by Critic")
    nomination_query = """
        SELECT c.critic_name, COUNT(g.id) AS nomination_count
        FROM games g
        JOIN critics c ON g.nominated_by = c.id
        GROUP BY c.critic_name ORDER BY nomination_count DESC;
    """
    nomination_df = conn.query(nomination_query)
    fig = go.Figure(data=[go.Pie(labels=nomination_df['critic_name'], values=nomination_df['nomination_count'], hole=.3)])
    fig.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Score Distribution by Critic (0-10)")
    binned_ratings_query = """
        SELECT 
            c.critic_name,
            FLOOR(r.score) as score_bin,
            COUNT(r.id) as rating_count
        FROM ratings r
        JOIN critics c ON r.critic_id = c.id
        WHERE r.score IS NOT NULL
        GROUP BY c.critic_name, score_bin;
    """
    binned_df = conn.query(binned_ratings_query)
    if not binned_df.empty:
        binned_pivot = binned_df.pivot(index='score_bin', columns='critic_name', values='rating_count').fillna(0)
        st.bar_chart(binned_pivot)

# --- Critic Controversy ---
st.header("Controversial Critic Analysis")
st.write("A 'contrarian' critic is one whose ratings deviate most from the average score for each game.")
controversy_query = """
    WITH game_avg AS (
        SELECT game_id, AVG(score) as avg_game_score
        FROM ratings WHERE score IS NOT NULL GROUP BY game_id
    )
    SELECT
        c.critic_name,
        AVG(ABS(r.score - ga.avg_game_score)) as controversy_score
    FROM ratings r
    JOIN critics c ON r.critic_id = c.id
    JOIN game_avg ga ON r.game_id = ga.game_id
    WHERE r.score IS NOT NULL
    GROUP BY c.critic_name
    ORDER BY controversy_score DESC;
"""
critic_controversy_df = conn.query(controversy_query)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Most Contrarian Critics")
    st.dataframe(critic_controversy_df.head(5), hide_index=True)
with col2:
    st.subheader("Least Contrarian Critics")
    st.dataframe(critic_controversy_df.sort_values("controversy_score", ascending=True).head(5), hide_index=True)