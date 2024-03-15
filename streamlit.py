import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Memperkenalkan aplikasi
st.title('Aplikasi Visualisasi Data dengan Streamlit')
st.write('Ini adalah contoh aplikasi sederhana untuk visualisasi data dengan Streamlit.')


# Membuat data acak untuk visualisasi
data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon']
)

# Plot garis
r_data = np.random.rand(10, 10)
st.subheader('Plot Garis')
st.line_chart(r_data)

# Plot scatter
st.subheader('Plot Scatter')
st.map(data)

# Histogram
r_data = np.random.rand(10, 10)
st.subheader('Histogram')
hist_values = np.histogram(
    data['lat'], bins=20, range=(-37.8, -37.7)
)[0]
#st.bar_chart(hist_values)
st.bar_chart(r_data)

# Heatmap
st.subheader('Heatmap')
# Create a heatmap using Seaborn
heatmap_data = np.random.rand(10, 10)
heatmap = sns.heatmap(heatmap_data, annot=True)
st.pyplot(heatmap.figure)  # Display the heatmap using Streamlit