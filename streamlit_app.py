# streamlit_app.py
import streamlit as st
import time

st.set_page_config(page_title="Login", layout="centered")

# --- Login Logic (remains the same) ---
def check_password():
    """Returns `True` if the password is correct."""
    return st.session_state.get("password") == st.secrets["password"]

# --- Page Navigation and Setup ---
if st.session_state.get("password_correct", False):
    st.toast("Login successful!")
    
    # Define all your pages with the new filenames
    descriptive_page = st.Page("pages/dashboard.py", title="Dashboard", default=True)
    # predictions_page = st.Page("pages/predictions.py", title="Predictions")
    analysis_page = st.Page("pages/analysis.py", title="Analysis")

    # The order in this list determines the sidebar order
    pg = st.navigation([descriptive_page, 
                        # predictions_page, 
                        analysis_page])
    
    pg.run()
    
else:
    # User is not logged in, show the login form.
    st.title("Private Dashboard Login")
    
    password_input = st.text_input(
        "Password", type="password", key="password"
    )

    if st.button("Sign In"):
        with st.spinner("Authenticating..."):
            time.sleep(0.5)
            if check_password():
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Password incorrect.")