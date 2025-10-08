# dashboard.py
import streamlit as st

# --- Authentication Check ---
# This must be at the top of every protected page
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Content ---
# The rest of your dashboard code goes here
st.set_page_config(page_title="My Dashboard")

st.title("Welcome to the Dashboard!")
st.write("This is your protected dashboard content.")

if st.button("Log out"):
    del st.session_state["password_correct"]
    st.rerun()

st.info("There is currently no data to display.")