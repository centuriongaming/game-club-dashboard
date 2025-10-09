# pages/predictions.py
import streamlit as st

# --- Authentication Check ---
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(page_title="Predictions")
st.title("Predictive Analytics")

# --- Placeholder Content ---
st.info("This section is under construction. Predictive modeling features will be available here soon!", icon="🚧")

st.write("Future features will include:")
st.markdown("""
-   Predicting a score for any critic on any game (including upcoming ones).
-   Ranking the features that contribute to each prediction.
-   Forecasting the likelihood a critic will skip a particular game.
""")