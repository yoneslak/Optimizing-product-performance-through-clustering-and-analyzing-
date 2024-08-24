import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV

# Create a sample DataFrame (replace with your own data)
df = pd.DataFrame({
    'soil_pH': np.random.normal(6.5, 0.5, 100),
    'nitrogen': np.random.normal(100, 20, 100),
    'phosphorus': np.random.normal(50, 10, 100),
    'potassium': np.random.normal(200, 30, 100),
    'temperature': np.random.normal(20, 2, 100),
    'precipitation': np.random.normal(50, 10, 100),  # mm
    'solar_radiation': np.random.normal(20, 5, 100),  # MJ/m²
    'wind_speed': np.random.normal(5, 2, 100),  # m/s
    'crop_yield': np.random.normal(500, 100, 100)
})

# Data Preparation
df.dropna(inplace=True)  # handle missing values
scaler = StandardScaler()
df[['soil_pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'precipitation', 'solar_radiation', 'wind_speed']] = scaler.fit_transform(df[['soil_pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'precipitation', 'solar_radiation', 'wind_speed']])

# Calculate mean and standard deviation of soil pH levels
pH_mean = np.mean(df['soil_pH'])
pH_std = np.std(df['soil_pH'])

# Calculate mean and standard deviation of NPK levels
N_mean = np.mean(df['nitrogen'])
P_mean = np.mean(df['phosphorus'])
K_mean = np.mean(df['potassium'])

# Calculate mean and standard deviation of temperature
temp_mean = np.mean(df['temperature'])
temp_std = np.std(df['temperature'])

# Calculate mean and standard deviation of precipitation
precip_mean = np.mean(df['precipitation'])
precip_std = np.std(df['precipitation'])

# Calculate mean and standard deviation of solar radiation
solar_mean = np.mean(df['solar_radiation'])
solar_std = np.std(df['solar_radiation'])

# Calculate mean and standard deviation of wind speed
wind_mean = np.mean(df['wind_speed'])
wind_std = np.std(df['wind_speed'])

# Create a heatmap to show correlation between factors
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', square=True)
plt.show()

# Create a scatter plot to visualize relationship between soil pH and crop yields
sns.scatterplot(x='soil_pH', y='crop_yield', data=df)
plt.show()

# Perform k-means clustering on the data
kmeans = KMeans(n_clusters=5)
kmeans.fit(df[['soil_pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'precipitation', 'solar_radiation', 'wind_speed']])

# Get the cluster labels for each data point
labels = kmeans.labels_

# Evaluate the clustering model using various metrics
silhouette = silhouette_score(df[['soil_pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'precipitation', 'solar_radiation', 'wind_speed']], labels)
calinski = calinski_harabasz_score(df[['soil_pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'precipitation', 'solar_radiation', 'wind_speed']], labels)
davies = davies_bouldin_score(df[['soil_pH', 'nitrogen', 'phosphorus', 'potassium', 'temperature', 'precipitation', 'solar_radiation', 'wind_speed']], labels)

print("Clustering Evaluation Metrics:")
print(f"  Silhouette Score: {silhouette:.3f}")
print(f"  Calinski-Harabasz Index: {calinski:.3f}")
print(f"  Davies-Bouldin Index: {davies:.3f}")

# Analyze the characteristics of each cluster
for i in range(5):
    cluster_df = df[labels == i]
    print(f"Cluster {i}:")
    print(f"  Mean soil pH: {np.mean(cluster_df['soil_pH'])}")
    print(f"  Mean NPK levels: {np.mean(cluster_df['nitrogen'])}, {np.mean(cluster_df['phosphorus'])}, {np.mean(cluster_df['potassium'])}")
    print(f"  Mean temperature: {np.mean(cluster_df['temperature'])}")
    print(f"  Mean precipitation: {np.mean(cluster_df['precipitation'])}")
    print(f"  Mean solar radiation: {np.mean(cluster_df['solar_radiation'])}")
    print(f"  Mean wind speed: {np.mean(cluster_df['wind_speed'])}")
    print(f"  Mean crop yield: {np.mean(cluster_df['crop_yield'])}")
    print()

# Identify the optimal locations to plant sunflower seeds
optimal_locations = df[labels == np.argmin([np.mean(df[labels == i]['crop_yield']) for i in range(5)])]
print("Optimal locations to plant sunflower seeds:")
print(optimal_locations)
#کد تجزیه و تحلیل داده ای برای کشت تخمه آفتابی با یادگیری ماشین