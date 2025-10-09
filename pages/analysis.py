import streamlit as st
import pandas as pd

# --- Authentication Check (Add to the top of the page) ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Descriptive Analytics")
st.title("Descriptive Analytics")
st.write("This page shows high-level statistics based on the raw ratings data.")

# Connect to the database
conn = st.connection("mydb", type="sql")

# --- 1. Critic Participation ---
st.header("Critic Participation")
st.write("This table shows how many games each critic has rated out of the total available.")
critic_participation_query = """
    SELECT
        c.critic_name,
        COUNT(r.score) AS ratings_given,
        (SELECT COUNT(*) FROM games) AS total_games,
        AVG(r.score) AS average_score
    FROM critics c
    LEFT JOIN ratings r ON c.critic_id = r.critic_id
    GROUP BY c.critic_name
    ORDER BY ratings_given DESC;
"""
critic_participation_df = conn.query(critic_participation_query)
critic_participation_df['participation_rate'] = critic_participation_df['ratings_given'] / critic_participation_df['total_games']
st.dataframe(
    critic_participation_df,
    column_config={
        "average_score": st.column_config.ProgressColumn(
            "Average Score",
            format="%.2f",
            min_value=0,
            max_value=10,
        ),
        "participation_rate": st.column_config.ProgressColumn(
            "Participation Rate",
            format="%.1f%%",
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True
)


# --- 2. Nomination Stats ---
st.header("Nomination Stats")
st.write("This table shows who has nominated the most games for review.")
nomination_df = conn.query("SELECT nominated_by, COUNT(*) AS count FROM games GROUP BY nominated_by ORDER BY count DESC;")
st.dataframe(nomination_df, hide_index=True)


# --- 3. Overall Score Distribution ---
st.header("Overall Score Distribution")
scores_df = conn.query("SELECT score FROM ratings WHERE score IS NOT NULL;")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Ratings Given", scores_df['score'].count())
    st.metric("Overall Average Score", f"{scores_df['score'].mean():.2f}")

with col2:
    st.subheader("Frequency of Scores")
    # Ensure scores are treated as a categorical variable for accurate counting
    score_counts = scores_df['score'].astype(str).value_counts().sort_index()
    st.bar_chart(score_counts)