# pages/1_Dashboard.py
import streamlit as st

# Set the page configuration at the top of the script
# This command must be the first Streamlit command used on a page.
st.set_page_config(page_title="My Dashboard", page_icon="📊")

def show_dashboard_content():
    """Contains the actual content of the dashboard page."""
    st.title("🚀 Welcome to the Dashboard!")
    st.write("This is your empty dashboard, ready for future data and visualizations.")

    st.info("There is currently no data to display.", icon="ℹ️")

    st.markdown("---")
    st.subheader("Placeholder for a Chart")
    st.write("A chart or graph will be displayed here.")

    st.subheader("Placeholder for Metrics")
    st.write("Key performance indicators (KPIs) will be shown here.")


# --- Main Page Logic ---

# Check if the user is authenticated
if st.session_state.get("password_correct", False):
    # If authenticated, show the dashboard content
    show_dashboard_content()
else:
    # If not authenticated, show an error message and a link to the login page
    st.error("You are not logged in. Please go to the main page to log in.")
    # Assuming your main app file is named 'streamlit_app.py'
    st.page_link("streamlit_app.py", label="Go to Login Page", icon="🏠")