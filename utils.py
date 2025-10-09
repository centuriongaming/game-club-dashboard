# utils.py
import streamlit as st

def check_auth():
    """
    Checks if the user is authenticated. If not, stops the app execution
    and displays an error message.
    """
    if not st.session_state.get("password_correct", False):
        st.error("🔒 You need to log in first.")
        st.stop()

@st.cache_resource
def get_db_connection():
    """
    Returns a cached database connection object. Caching ensures we don't
    reconnect to the DB on every page interaction.
    """
    try:
        return st.connection("mydb", type="sql")
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()