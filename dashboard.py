# dashboard.py
import streamlit as st

# --- Authentication Check ---
# This must be at the top of every protected page
if not st.session_state.get("password_correct", False):
    st.error("You need to log in first.")
    st.stop()

# --- Page Content ---
st.set_page_config(page_title="My Dashboard")
st.title("Welcome to the Dashboard!")

# --- Database Connection Test ---
with st.status("Connecting to database...", expanded=True) as status:
    try:
        conn = st.connection("mydb", type="sql")
        # Perform a simple query to verify the connection
        df = conn.query("SELECT 1 as test_col;")
        
        # Check if the query returned the expected result
        if not df.empty and df['test_col'][0] == 1:
            status.update(label="Connection successful!", state="complete", expanded=False)
            st.success("You are connected to the database.")
        else:
            status.update(label="Connection test failed!", state="error", expanded=True)

    except Exception as e:
        status.update(label=f"Connection failed: {e}", state="error", expanded=True)


# --- Rest of the Dashboard ---
if st.button("Log out"):
    del st.session_state["password_correct"]
    st.rerun()

st.info("There is currently no data to display.")