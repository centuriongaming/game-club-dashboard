# pages/1_Dashboard.py
import streamlit as st

# Verify the user has logged in.
if not st.session_state.get("password_correct", False):
    st.error("You must log in first on the main page.")
    st.stop()

# --- Dashboard Content ---
# This code will only run if the password was correct.

st.set_page_config(page_title="My Dashboard", page_icon="📊")

st.title("🚀 Welcome to the Dashboard!")
st.write("This is your empty dashboard, ready for future data and visualizations.")

st.info("There is currently no data to display.", icon="ℹ️")

st.markdown("---")
st.subheader("Placeholder for a Chart")
st.write("A chart or graph will be displayed here.")

st.subheader("Placeholder for Metrics")
st.write("Key performance indicators (KPIs) will be shown here.")