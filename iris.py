import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
df['Species'] = df['Species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Streamlit app title
st.title(" :seedling: Iris Dataset EDA")

# Sidebar for filters
st.sidebar.header("Choose your filter: ")

# Filter by Species
species = st.sidebar.multiselect("Pick the Species", df["Species"].unique())
if not species:
    filtered_df = df.copy()
else:
    filtered_df = df[df["Species"].isin(species)]

# Display filtered data
st.subheader("Filtered Data")
st.write(filtered_df)

# Visualizations
st.subheader("Visualizations")

# Pairplot for Iris dataset
st.write("Pairplot of Iris Dataset")
fig = px.scatter_matrix(filtered_df, dimensions=iris.feature_names, color="Species")
st.plotly_chart(fig, use_container_width=True)

# Boxplot for each feature by species
st.write("Boxplot of Features by Species")
for feature in iris.feature_names:
    fig = px.box(filtered_df, x="Species", y=feature, color="Species")
    st.plotly_chart(fig, use_container_width=True)

# Scatter plot for sepal length vs sepal width
st.write("Scatter Plot: Sepal Length vs Sepal Width")
fig = px.scatter(filtered_df, x="sepal length (cm)", y="sepal width (cm)", color="Species")
st.plotly_chart(fig, use_container_width=True)

# Scatter plot for petal length vs petal width
st.write("Scatter Plot: Petal Length vs Petal Width")
fig = px.scatter(filtered_df, x="petal length (cm)", y="petal width (cm)", color="Species")
st.plotly_chart(fig, use_container_width=True)

# Download filtered data
st.subheader("Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name="filtered_iris.csv",
    mime="text/csv",
    help="Click here to download the filtered data as a CSV file."
)
