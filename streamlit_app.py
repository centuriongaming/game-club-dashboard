# app.py
import streamlit as st

# Function to check the password
def check_password():
    """Returns `True` if the user entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # The password is set in the Streamlit secrets
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Initialize the session state if it's not already done
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    # Display the password input field
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    
    # Give a hint if the password is wrong
    if not st.session_state["password_correct"]:
        st.write("Please enter the password to proceed.")
        
    return st.session_state["password_correct"]


# --- Main Application ---

# Set the page configuration
st.set_page_config(page_title="My Dashboard", page_icon="🔒")

# Check if the password is correct
if not check_password():
    st.stop() # Stop the app from running further if the password is not correct

# If the password is correct, display the dashboard
st.title("🚀 Welcome to the Dashboard!")
st.write("This is your empty dashboard, ready for future data and visualizations.")

st.info("There is currently no data to display.", icon="ℹ️")

st.markdown("---")
st.subheader("Placeholder for a Chart")
st.write("A chart or graph will be displayed here.")

st.subheader("Placeholder for Metrics")
st.write("Key performance indicators (KPIs) will be shown here.")