# utils.py
import json
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

@st.cache_data
def load_queries():
    """Loads queries from the JSON file and joins multi-line queries."""
    with open('queries.json', 'r') as f:
        queries = json.load(f)
    # Join list of strings into a single string for each query
    for key, value in queries.items():
        if isinstance(value, list):
            queries[key] = " ".join(value)
    return queries