# streamlit_app.py
import streamlit as st
import time

st.set_page_config(page_title="Login", layout="centered")

# --- Login Logic ---
def check_password():
    """Returns `True` if the password is correct."""
    return st.session_state.get("password") == st.secrets["password"]

# --- Navigation and Page Setup ---
if st.session_state.get("password_correct", False):
    # User is logged in, so we'll register the dashboard page.
    st.toast("Login successful!")
    dashboard_page = st.Page(
        "dashboard.py", title="Main Dashboard"
    )
    st.navigation([dashboard_page]).run()
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