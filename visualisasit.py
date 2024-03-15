import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a title and subheader
st.title('Interactive Data Visualization with Streamlit')
st.subheader('Exploring different types of interactive visualizations')

# Create a dataframe for demonstration
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

# Line plot
st.subheader('Line Plot')
st.line_chart(data)

# Scatter plot
st.subheader('Scatter Plot')
st.write(data)

# Histogram
st.subheader('Histogram')
st.write(data)
plt.hist(data['x'], bins=20)
st.pyplot(plt.gcf())

# Heatmap
# Heatmap using Seaborn
st.subheader('Heatmap')
heatmap_data = np.random.rand(10, 10)
heatmap_fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, ax=ax)
st.pyplot(heatmap_fig)