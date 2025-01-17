# Function.py
# Initialize session state
def initialize_session_state():
    if "active_section" not in st.session_state:
        st.session_state.active_section = None
    if "graphs" not in st.session_state:
        st.session_state.graphs = []  # Store generated graphs
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()  # Store the DataFrame
    if "model" not in st.session_state:
        st.session_state.model = None  # Store the selected model
    if "predictions" not in st.session_state:
        st.session_state.predictions = None  # Store the predictions

# Sidebar navigation
def sidebar_navigation():
    st.sidebar.title("Data Analysis Tasks")
    st.sidebar.header("Navigation")
    if st.sidebar.button("Import Data"):
        st.session_state.active_section = "import_data"
        st.session_state.graphs = []  # Reset graphs when switching sections
    if st.sidebar.button("Missing Values"):
        st.session_state.active_section = "missing_values"
        st.session_state.graphs = []  # Reset graphs when switching sections
    if st.sidebar.button("Data Visualization"):
        st.session_state.active_section = "data_visualization"
    if st.sidebar.button("Advanced Data Visualization"):
        st.session_state.active_section = "advanced_data_visualization"
        st.session_state.graphs = []  # Reset graphs when switching sections
    if st.sidebar.button("Machine Learning"):
        st.session_state.active_section = "machine_learning"
        st.session_state.graphs = []  # Reset graphs when switching sections

# Import data section
def import_data_section():
    st.write("### Import Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data successfully loaded!")
        st.write("### Dataset")
        st.write(df)
        st.session_state.df = df  # Store the DataFrame in session state

# Handle missing values section
def missing_values_section():
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        st.write("### Handle Missing Values")
        st.write("#### NaN Values Summary")
        nan_sum = df.isnull().sum()  # Calculate the sum of NaN values for each column
        st.write(nan_sum)  # Display the sum of NaN values

        st.write("#### Replace NaN Values")
        for column in df.columns:
            if df[column].isnull().any():  # Only show options for columns with NaN values
                st.write(f"**Column: {column}**")
                replace_option = st.selectbox(
                    f"How to replace NaN in '{column}'?",
                    options=["Do not replace", "Replace with mean", "Replace with median", "Replace with mode", "Replace with custom value"],
                    key=column
                )

                if replace_option == "Replace with mean":
                    if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column is numeric
                        df[column].fillna(df[column].mean(), inplace=True)
                    else:
                        st.warning(f"Cannot calculate mean for non-numeric column: {column}")
                elif replace_option == "Replace with median":
                    if pd.api.types.is_numeric_dtype(df[column]):  # Check if the column is numeric
                        df[column].fillna(df[column].median(), inplace=True)
                    else:
                        st.warning(f"Cannot calculate median for non-numeric column: {column}")
                elif replace_option == "Replace with mode":
                    df[column].fillna(df[column].mode()[0], inplace=True)  # Mode works for both numeric and non-numeric columns
                elif replace_option == "Replace with custom value":
                    custom_value = st.text_input(f"Enter custom value for '{column}'", key=f"custom_{column}")
                    if custom_value:
                        df[column].fillna(custom_value, inplace=True)

        if st.button("Apply NaN Replacements"):
            st.success("NaN values have been replaced!")
            st.session_state.df = df  # Update the DataFrame in session state
            st.write("### Updated Dataset")
            st.write(df)
    else:
        st.warning("Please import data first.")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_labels(graph_type, **kwargs):
    """Generate titles and labels for graphs based on the graph type."""
    if graph_type == "Bar Plot":
        title = f"Bar Plot: {kwargs['x_axis']} vs {kwargs['y_axis']}"
        x_label = kwargs['x_axis']
        y_label = kwargs['y_axis']
    elif graph_type == "Line Plot":
        title = f"Line Plot: {kwargs['x_axis']} vs {kwargs['y_axis']}"
        x_label = kwargs['x_axis']
        y_label = kwargs['y_axis']
    elif graph_type == "Scatter Plot":
        title = f"Scatter Plot: {kwargs['x_axis']} vs {kwargs['y_axis']}"
        x_label = kwargs['x_axis']
        y_label = kwargs['y_axis']
    elif graph_type == "Histogram":
        title = f"Histogram of {kwargs['column']}"
        x_label = kwargs['column']
        y_label = "Frequency"
    elif graph_type == "Box Plot":
        title = f"Box Plot of {kwargs['column']}"
        x_label = kwargs['column']
        y_label = "Values"
    elif graph_type == "Pie Chart":
        title = f"Pie Chart of {kwargs['column']}"
        x_label = None
        y_label = None
    elif graph_type == "Area Plot":
        title = f"Area Plot: {kwargs['x_axis']} vs {kwargs['y_axis']}"
        x_label = kwargs['x_axis']
        y_label = kwargs['y_axis']
    elif graph_type == "Violin Plot":
        title = f"Violin Plot of {kwargs['x_axis']} vs {kwargs['y_axis']}"
        x_label = kwargs['x_axis']
        y_label = kwargs['y_axis']
    elif graph_type == "Heatmap":
        title = "Heatmap of Correlation Matrix"
        x_label = None
        y_label = None
    elif graph_type == "Pair Plot":
        title = "Pair Plot of DataFrame"
        x_label = None
        y_label = None
    elif graph_type == "Count Plot":
        title = f"Count Plot of {kwargs['x_axis']}"
        x_label = kwargs['x_axis']
        y_label = "Count"
    elif graph_type == "Swarm Plot":
        title = f"Swarm Plot of {kwargs['x_axis']} vs {kwargs['y_axis']}"
        x_label = kwargs['x_axis']
        y_label = kwargs['y_axis']
    elif graph_type == "KDE Plot":
        title = f"KDE Plot of {kwargs['column']}"
        x_label = kwargs['column']
        y_label = "Density"
    elif graph_type == "Rug Plot":
        title = f"Rug Plot of {kwargs['column']}"
        x_label = kwargs['column']
        y_label = None
    return title, x_label, y_label

def generate_code_snippet(graph_type, **kwargs):
    """Generate the code snippet for the selected graph type."""
    if graph_type == "Bar Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.barplot(x=df['{kwargs['x_axis']}'], y=df['{kwargs['y_axis']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Line Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.lineplot(x=df['{kwargs['x_axis']}'], y=df['{kwargs['y_axis']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Scatter Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.scatterplot(x=df['{kwargs['x_axis']}'], y=df['{kwargs['y_axis']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Histogram":
        code = f"""
        fig, ax = plt.subplots()
        sns.histplot(df['{kwargs['column']}'], ax=ax, color='{kwargs['color']}', kde=True)
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Box Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.boxplot(x=df['{kwargs['column']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Pie Chart":
        code = f"""
        fig, ax = plt.subplots()
        df['{kwargs['column']}'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['{kwargs['color']}'])
        ax.set_title('{kwargs['title']}')
        """
    elif graph_type == "Area Plot":
        code = f"""
        fig, ax = plt.subplots()
        df.plot(kind='area', x='{kwargs['x_axis']}', y='{kwargs['y_axis']}', ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Violin Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.violinplot(x=df['{kwargs['x_axis']}'], y=df['{kwargs['y_axis']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Heatmap":
        code = """
        fig, ax = plt.subplots()
        sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Heatmap of Correlation Matrix')
        """
    elif graph_type == "Pair Plot":
        code = """
        fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
        fig.fig.suptitle('Pair Plot of DataFrame')
        """
    elif graph_type == "Count Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.countplot(x=df['{kwargs['x_axis']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Swarm Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.swarmplot(x=df['{kwargs['x_axis']}'], y=df['{kwargs['y_axis']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "KDE Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.kdeplot(df['{kwargs['column']}'], ax=ax, color='{kwargs['color']}', fill=True)
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        ax.set_ylabel('{kwargs['y_label']}')
        """
    elif graph_type == "Rug Plot":
        code = f"""
        fig, ax = plt.subplots()
        sns.rugplot(df['{kwargs['column']}'], ax=ax, color='{kwargs['color']}')
        ax.set_title('{kwargs['title']}')
        ax.set_xlabel('{kwargs['x_label']}')
        """
    return code

def data_visualization_section():
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        st.write("### Data Visualization")

        # Graph type selection
        graph_type = st.selectbox(
            "Select the type of graph",
            options=[
                "Bar Plot", "Line Plot", "Scatter Plot", "Histogram", "Box Plot",
                "Pie Chart", "Area Plot", "Violin Plot", "Heatmap", "Pair Plot",
                "Count Plot", "Swarm Plot", "KDE Plot", "Rug Plot"
            ],
            key="graph_type"
        )

        # Color selection
        color = st.color_picker("Choose a color for the graph", "#1f77b4", key="graph_color")

        # Placeholder for graph and code
        graph_placeholder = st.empty()
        code_placeholder = st.empty()

        # Generate graph based on selected type
        if graph_type in ["Bar Plot", "Line Plot", "Scatter Plot", "Area Plot", "Violin Plot", "Swarm Plot"]:
            x_axis = st.selectbox("Select X-axis column", df.columns, key=f"{graph_type.lower().replace(' ', '_')}_x")
            y_axis = st.selectbox("Select Y-axis column", df.columns, key=f"{graph_type.lower().replace(' ', '_')}_y")
            if st.button(f"Generate {graph_type}"):
                try:
                    title, x_label, y_label = generate_labels(graph_type, x_axis=x_axis, y_axis=y_axis)
                    fig, ax = plt.subplots()
                    if graph_type == "Bar Plot":
                        sns.barplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    elif graph_type == "Line Plot":
                        sns.lineplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    elif graph_type == "Scatter Plot":
                        sns.scatterplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    elif graph_type == "Area Plot":
                        df.plot(kind="area", x=x_axis, y=y_axis, ax=ax, color=color)
                    elif graph_type == "Violin Plot":
                        sns.violinplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    elif graph_type == "Swarm Plot":
                        sns.swarmplot(x=df[x_axis], y=df[y_axis], ax=ax, color=color)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    graph_placeholder.pyplot(fig)
                    code_placeholder.code(generate_code_snippet(graph_type, x_axis=x_axis, y_axis=y_axis, color=color, title=title, x_label=x_label, y_label=y_label), language="python")
                except KeyError as e:
                    st.error(f"Column not found in DataFrame: {e}")

        elif graph_type in ["Histogram", "Box Plot", "Pie Chart", "KDE Plot", "Rug Plot"]:
            column = st.selectbox("Select column", df.columns, key=f"{graph_type.lower().replace(' ', '_')}_col")
            if st.button(f"Generate {graph_type}"):
                try:
                    title, x_label, y_label = generate_labels(graph_type, column=column)
                    fig, ax = plt.subplots()
                    if graph_type == "Histogram":
                        sns.histplot(df[column], ax=ax, color=color, kde=True)
                    elif graph_type == "Box Plot":
                        sns.boxplot(x=df[column], ax=ax, color=color)
                    elif graph_type == "Pie Chart":
                        df[column].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=[color])
                    elif graph_type == "KDE Plot":
                        sns.kdeplot(df[column], ax=ax, color=color, fill=True)
                    elif graph_type == "Rug Plot":
                        sns.rugplot(df[column], ax=ax, color=color)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    if y_label:
                        ax.set_ylabel(y_label)
                    graph_placeholder.pyplot(fig)
                    code_placeholder.code(generate_code_snippet(graph_type, column=column, color=color, title=title, x_label=x_label, y_label=y_label), language="python")
                except KeyError as e:
                    st.error(f"Column not found in DataFrame: {e}")

        elif graph_type == "Heatmap":
            if st.button("Generate Heatmap"):
                try:
                    title, _, _ = generate_labels(graph_type)
                    fig, ax = plt.subplots()
                    # Filter numerical columns only
                    numerical_df = df.select_dtypes(include=['float64', 'int64'])
                    if numerical_df.empty:
                        st.error("No numerical columns found for heatmap.")
                    else:
                        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                        ax.set_title(title)
                        graph_placeholder.pyplot(fig)
                        code_placeholder.code(generate_code_snippet(graph_type), language="python")
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")

        elif graph_type == "Pair Plot":
            if st.button("Generate Pair Plot"):
                try:
                    title, _, _ = generate_labels(graph_type)
                    # Filter numerical columns only
                    numerical_df = df.select_dtypes(include=['float64', 'int64'])
                    if numerical_df.empty:
                        st.error("No numerical columns found for pair plot.")
                    else:
                        fig = sns.pairplot(numerical_df)
                        fig.fig.suptitle(title)
                        graph_placeholder.pyplot(fig)
                        code_placeholder.code(generate_code_snippet(graph_type), language="python")
                except Exception as e:
                    st.error(f"Error generating pair plot: {e}")

        elif graph_type == "Count Plot":
            x_axis = st.selectbox("Select X-axis column", df.columns, key="count_x")
            if st.button("Generate Count Plot"):
                try:
                    title, x_label, y_label = generate_labels(graph_type, x_axis=x_axis)
                    fig, ax = plt.subplots()
                    sns.countplot(x=df[x_axis], ax=ax, color=color)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    graph_placeholder.pyplot(fig)
                    code_placeholder.code(generate_code_snippet(graph_type, x_axis=x_axis, color=color, title=title, x_label=x_label, y_label=y_label), language="python")
                except KeyError as e:
                    st.error(f"Column not found in DataFrame: {e}")

    else:
        st.warning("Please import data first.")

# Advanced data visualization section
def advanced_data_visualization_section():
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        st.write("### Advanced Data Visualization")

        # Choose between entire dataset or top 10 values
        data_scope = st.radio(
            "Select data scope",
            options=["Entire Dataset", "Top 10 Values"],
            key="data_scope"
        )

        # Graph type selection
        graph_type = st.selectbox(
            "Select the type of graph",
            options=[
                "Bar Plot", "Line Plot", "Scatter Plot", "Histogram", "Box Plot",
                "Pie Chart", "Area Plot", "Violin Plot", "Heatmap", "Pair Plot",
                "Count Plot", "Swarm Plot", "KDE Plot", "Rug Plot"
            ],
            key="advanced_graph_type"
        )

        # Color selection
        color = st.color_picker("Choose a color for the graph", "#1f77b4", key="graph_color")

        # Filter data based on selection
        if data_scope == "Top 10 Values":
            column = st.selectbox("Select column for top 10 values", df.columns, key="top_values_col")
            top_values = df[column].value_counts().nlargest(10)  # Get top 10 values
            df_filtered = df[df[column].isin(top_values.index)]  # Filter rows with top 10 values
        else:
            df_filtered = df  # Use entire dataset

        # Placeholder for graph and code
        graph_placeholder = st.empty()
        code_placeholder = st.empty()

        # Generate graph based on selected type
        if graph_type in ["Bar Plot", "Line Plot", "Scatter Plot", "Area Plot", "Violin Plot", "Swarm Plot"]:
            x_axis = st.selectbox("Select X-axis column", df_filtered.columns, key=f"{graph_type.lower().replace(' ', '_')}_x")
            y_axis = st.selectbox("Select Y-axis column", df_filtered.columns, key=f"{graph_type.lower().replace(' ', '_')}_y")
            if st.button(f"Generate {graph_type}"):
                try:
                    title, x_label, y_label = generate_labels(graph_type, x_axis=x_axis, y_axis=y_axis)
                    fig, ax = plt.subplots()
                    if graph_type == "Bar Plot":
                        sns.barplot(x=df_filtered[x_axis], y=df_filtered[y_axis], ax=ax, color=color)
                    elif graph_type == "Line Plot":
                        sns.lineplot(x=df_filtered[x_axis], y=df_filtered[y_axis], ax=ax, color=color)
                    elif graph_type == "Scatter Plot":
                        sns.scatterplot(x=df_filtered[x_axis], y=df_filtered[y_axis], ax=ax, color=color)
                    elif graph_type == "Area Plot":
                        df_filtered.plot(kind="area", x=x_axis, y=y_axis, ax=ax, color=color)
                    elif graph_type == "Violin Plot":
                        sns.violinplot(x=df_filtered[x_axis], y=df_filtered[y_axis], ax=ax, color=color)
                    elif graph_type == "Swarm Plot":
                        sns.swarmplot(x=df_filtered[x_axis], y=df_filtered[y_axis], ax=ax, color=color)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    graph_placeholder.pyplot(fig)
                    code_placeholder.code(generate_code_snippet(graph_type, x_axis=x_axis, y_axis=y_axis, color=color, title=title, x_label=x_label, y_label=y_label), language="python")
                except KeyError as e:
                    st.error(f"Column not found in DataFrame: {e}")

        elif graph_type in ["Histogram", "Box Plot", "Pie Chart", "KDE Plot", "Rug Plot"]:
            column = st.selectbox("Select column", df_filtered.columns, key=f"{graph_type.lower().replace(' ', '_')}_col")
            if st.button(f"Generate {graph_type}"):
                try:
                    title, x_label, y_label = generate_labels(graph_type, column=column)
                    fig, ax = plt.subplots()
                    if graph_type == "Histogram":
                        sns.histplot(df_filtered[column], ax=ax, color=color, kde=True)
                    elif graph_type == "Box Plot":
                        sns.boxplot(x=df_filtered[column], ax=ax, color=color)
                    elif graph_type == "Pie Chart":
                        df_filtered[column].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax, colors=[color])
                    elif graph_type == "KDE Plot":
                        sns.kdeplot(df_filtered[column], ax=ax, color=color, fill=True)
                    elif graph_type == "Rug Plot":
                        sns.rugplot(df_filtered[column], ax=ax, color=color)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    if y_label:
                        ax.set_ylabel(y_label)
                    graph_placeholder.pyplot(fig)
                    code_placeholder.code(generate_code_snippet(graph_type, column=column, color=color, title=title, x_label=x_label, y_label=y_label), language="python")
                except KeyError as e:
                    st.error(f"Column not found in DataFrame: {e}")

        elif graph_type == "Heatmap":
            if st.button("Generate Heatmap"):
                try:
                    title, _, _ = generate_labels(graph_type)
                    fig, ax = plt.subplots()
                    # Filter numerical columns only
                    numerical_df = df_filtered.select_dtypes(include=['float64', 'int64'])
                    if numerical_df.empty:
                        st.error("No numerical columns found for heatmap.")
                    else:
                        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                        ax.set_title(title)
                        graph_placeholder.pyplot(fig)
                        code_placeholder.code(generate_code_snippet(graph_type), language="python")
                except Exception as e:
                    st.error(f"Error generating heatmap: {e}")

        elif graph_type == "Pair Plot":
            if st.button("Generate Pair Plot"):
                try:
                    title, _, _ = generate_labels(graph_type)
                    # Filter numerical columns only
                    numerical_df = df_filtered.select_dtypes(include=['float64', 'int64'])
                    if numerical_df.empty:
                        st.error("No numerical columns found for pair plot.")
                    else:
                        fig = sns.pairplot(numerical_df)
                        fig.fig.suptitle(title)
                        graph_placeholder.pyplot(fig)
                        code_placeholder.code(generate_code_snippet(graph_type), language="python")
                except Exception as e:
                    st.error(f"Error generating pair plot: {e}")

        elif graph_type == "Count Plot":
            x_axis = st.selectbox("Select X-axis column", df_filtered.columns, key="count_x")
            if st.button("Generate Count Plot"):
                try:
                    title, x_label, y_label = generate_labels(graph_type, x_axis=x_axis)
                    fig, ax = plt.subplots()
                    sns.countplot(x=df_filtered[x_axis], ax=ax, color=color)
                    ax.set_title(title)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    graph_placeholder.pyplot(fig)
                    code_placeholder.code(generate_code_snippet(graph_type, x_axis=x_axis, color=color, title=title, x_label=x_label, y_label=y_label), language="python")
                except KeyError as e:
                    st.error(f"Column not found in DataFrame: {e}")

    else:
        st.warning("Please import data first.")
# Machine learning section
def machine_learning_section():
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        st.write("### Machine Learning")

        # Choose between supervised and unsupervised learning
        learning_type = st.selectbox(
            "Select the type of learning",
            options=["Supervised", "Unsupervised"],
            key="learning_type"
        )

        if learning_type == "Supervised":
            st.write("#### Supervised Learning")
            target_column = st.selectbox("Select target column", df.columns, key="target_column")
            feature_columns = st.multiselect("Select feature columns", df.columns, key="feature_columns")

            if target_column and feature_columns:
                X = df[feature_columns]
                y = df[target_column]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Choose a model
                model_type = st.selectbox(
                    "Select a model",
                    options=["Linear Regression", "Logistic Regression"],
                    key="model_type"
                )

                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Logistic Regression":
                    model = LogisticRegression()

                # Train the model
                if st.button("Train Model"):
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    st.success("Model trained successfully!")

                # Make predictions
                if st.session_state.model is not None:
                    st.write("#### Make Predictions")
                    input_data = {}
                    for column in feature_columns:
                        input_data[column] = st.number_input(f"Enter value for {column}", key=f"input_{column}")

                    if st.button("Predict"):
                        input_df = pd.DataFrame([input_data])
                        prediction = st.session_state.model.predict(input_df)
                        st.session_state.predictions = prediction
                        st.success(f"Prediction: {prediction[0]}")

        elif learning_type == "Unsupervised":
            st.write("#### Unsupervised Learning")
            feature_columns = st.multiselect("Select feature columns", df.columns, key="feature_columns")

            if feature_columns:
                X = df[feature_columns]

                # Choose a model
                model_type = st.selectbox(
                    "Select a model",
                    options=["K-Means Clustering"],
                    key="model_type"
                )

                if model_type == "K-Means Clustering":
                    n_clusters = st.number_input("Enter number of clusters", min_value=2, max_value=10, value=2, key="n_clusters")
                    model = KMeans(n_clusters=n_clusters)

                # Train the model
                if st.button("Train Model"):
                    model.fit(X)
                    st.session_state.model = model
                    st.success("Model trained successfully!")

                # Make predictions
                if st.session_state.model is not None:
                    st.write("#### Make Predictions")
                    input_data = {}
                    for column in feature_columns:
                        input_data[column] = st.number_input(f"Enter value for {column}", key=f"input_{column}")

                    if st.button("Predict"):
                        input_df = pd.DataFrame([input_data])
                        prediction = st.session_state.model.predict(input_df)
                        st.session_state.predictions = prediction
                        st.success(f"Prediction: Cluster {prediction[0]}")

    else:
        st.warning("Please import data first.")
