# streamlit_app.py
import streamlit as st
import time

st.set_page_config(page_title="Login", layout="centered")

# --- Login Logic ---
def check_password():
    """Returns `True` if the password is correct."""
    return st.session_state.get("password") == st.secrets["password"]

# --- Navigation and Page Setup ---
# Conditionally register the dashboard page
if st.session_state.get("password_correct", False):
    # The user is logged in, so show the dashboard page.
    # The title and icon are optional.
    dashboard_page = st.Page(
        "dashboard.py", title="Main Dashboard", page_icon="📊"
    )
    st.navigation([dashboard_page]).run()
else:
    # The user is not logged in, so show the login form.
    st.title("🔒 Private Dashboard Login")
    
    password_input = st.text_input(
        "Password", type="password", key="password"
    )

    if st.button("Sign In"):
        with st.spinner("Authenticating..."):
            time.sleep(0.5)
            if check_password():
                st.session_state["password_correct"] = True
                # Rerun the app to enter the logged-in state
                st.rerun()
            else:
                st.error("😕 Password incorrect.")