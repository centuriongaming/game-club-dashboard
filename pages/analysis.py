# pages/analysis.py
import streamlit as st
import pandas as pd

# --- Authentication Check (Add to the top of the page) ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

st.set_page_config(page_title="Analysis")
st.title("Analytical Rankings")
st.write("This page shows rankings based on the pre-calculated model results.")

# Connect to the database
try:
    conn = st.connection("mydb", type="sql")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# --- Query the pre-calculated results from the model_results table ---
st.header("Critic Analysis")
critic_analysis_query = """
    SELECT 
        c.critic_name,
        mr.adjusted_bias,
        mr.prediction_error
    FROM model_results mr
    JOIN critics c ON mr.critic_id = c.id;
"""
critic_analysis_df = conn.query(critic_analysis_query)

if critic_analysis_df.empty:
    st.warning("Model results have not been loaded into the database yet.")
else:
    tab1, tab2 = st.tabs(["Sentiment Analysis", "Alignment Analysis"])

    with tab1:
        st.subheader("Most Positive vs. Negative Reviewers")
        st.write("Based on the 'Adjusted Bias' calculated by the model.")
        # Sort and display the DataFrame queried from the database
        st.dataframe(
            critic_analysis_df[['critic_name', 'adjusted_bias']].sort_values("adjusted_bias", ascending=False),
            hide_index=True
        )

    with tab2:
        st.subheader("Most Aligned vs. Contrarian Reviewers")
        st.write("Based on the average 'Prediction Error' from the model.")
        # Sort and display the DataFrame queried from the database
        st.dataframe(
            critic_analysis_df[['critic_name', 'prediction_error']].sort_values("prediction_error", ascending=True),
            hide_index=True
        )

# --- You can add Item Analysis here as well ---
# st.header("Item Analysis")
# item_analysis_query = """ ... Query for item biases ... """
# ...