import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

from Function import initialize_session_state, sidebar_navigation,import_data_section,missing_values_section,generate_labels,generate_code_snippet,data_visualization_section,advanced_data_visualization_section,machine_learning_section
# Main app logic
def main():
    initialize_session_state()
    sidebar_navigation()

    # Display the active section
    if st.session_state.active_section == "import_data":
        import_data_section()
    elif st.session_state.active_section == "missing_values":
        missing_values_section()
    elif st.session_state.active_section == "data_visualization":
        data_visualization_section()
    elif st.session_state.active_section == "advanced_data_visualization":
        advanced_data_visualization_section()
    elif st.session_state.active_section == "machine_learning":
        machine_learning_section()
    else:
        st.write("Please select a task from the sidebar to get started.")

# Run the app
if __name__ == "__main__":
    main()