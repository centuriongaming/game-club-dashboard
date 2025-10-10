# utils.py
import json
import streamlit as st

def check_auth():
    """Checks if the user is authenticated."""
    if not st.session_state.get("password_correct", False):
        st.error("🔒 You need to log in first.")
        st.stop()

@st.cache_resource
def get_sqla_session():
    """Returns a cached SQLAlchemy session object."""
    return st.connection("mydb", type="sql").session

@st.cache_data
def load_queries():
    """Loads and joins raw SQL queries from the JSON file."""
    with open('queries.json', 'r') as f:
        queries = json.load(f)
    for key, value in queries.items():
        if isinstance(value, list):
            queries[key] = " ".join(value)
    return queries