# app.py
import streamlit as st
import time

st.set_page_config(page_title="Password Lock", page_icon="🔒")

# This function is now simpler, just checking the state.
def check_password_in_state():
    """Returns `True` if the password in session state is correct."""
    return st.session_state.get("password") == st.secrets["password"]

# --- Main App Logic ---
st.title("🔒 Private Dashboard Login")

# If user is already logged in, show success and the link.
if st.session_state.get("password_correct", False):
    st.success("Logged in successfully! 🎉")
    st.markdown("Please click the link below to go to your dashboard.")
    st.page_link("pages/dashboard.py", label="Go to Dashboard", icon="🚀")
    
# If not logged in, show the login form.
else:
    password_input = st.text_input(
        "Password", type="password", key="password"
    )

    if st.button("Sign In"):
        # Add a spinner for visual feedback (the animation)
        with st.spinner("Authenticating..."):            
            # Check the password
            if check_password_in_state():
                st.session_state["password_correct"] = True
                # Rerun the app to show the success message and dashboard link
                st.rerun()
            else:
                st.error("😕 Password incorrect. Please try again.")