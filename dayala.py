#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Ignore all warnings.
warnings.filterwarnings('ignore')

# Load the dataset
boston_data = pd.read_csv("BostonHousing.csv")

# Display the first few rows of the dataset
print(boston_data.head())

# Check for missing values
print(boston_data.isnull().sum())

# Summary statistics
print(boston_data.describe())

# Histogram of crime rate (crim)
plt.figure(figsize=(8, 6))
plt.hist(boston_data['crim'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Crime Rate')
plt.xlabel('Crime Rate')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of average number of rooms (rm) vs. median value of owner-occupied homes (medv)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rm', y='medv', data=boston_data, color='purple')
plt.title('Average Number of Rooms vs. Median Home Value')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value')
plt.show()

# Create the heatmap for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(boston_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



# Handle missing values (NaNs) by imputing the missing values with the mean of each column
from sklearn.impute import SimpleImputer

# Imputer instance
imputer = SimpleImputer(strategy='mean')
X = boston_data.drop('quality', axis=1)
# Impute missing values in the dataset
X_imputed = imputer.fit_transform(X)

# Data Preprocessing: Standardize features after imputation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Now continue with the rest of the code for clustering


# Determine the optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Fit k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
boston_data['Cluster'] = kmeans.labels_

# Visualize clustering results (scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rm', y='medv', hue='Cluster', data=boston_data, palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value')
plt.show()
# Histogram of cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=boston_data)
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Bar chart of cluster distribution
plt.figure(figsize=(8, 6))
boston_data['Cluster'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Pie chart of cluster distribution
plt.figure(figsize=(8, 8))
boston_data['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Pastel1'))
plt.title('Distribution of Clusters')
plt.ylabel('')
plt.show()

# Line graph
plt.figure(figsize=(8, 6))
plt.plot(boston_data['age'], color='blue')
plt.title('Age Distribution')
plt.xlabel('Index')
plt.ylabel('Age')
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(boston_data['Cluster'], kmeans.labels_)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Heatmap (alternative way)
plt.figure(figsize=(10, 8))
sns.heatmap(boston_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot (Corner plot)
sns.pairplot(boston_data)
plt.title('Pairplot')
plt.show()

# Box plot
plt.figure(figsize=(10, 8))
sns.boxplot(x='Cluster', y='medv', data=boston_data)
plt.title('Box Plot of Cluster vs. Median Home Value')
plt.xlabel('Cluster')
plt.ylabel('Median Home Value')
plt.show()

# Violin plot
plt.figure(figsize=(10, 8))
sns.violinplot(x='Cluster', y='medv', data=boston_data)
plt.title('Violin Plot of Cluster vs. Median Home Value')
plt.xlabel('Cluster')
plt.ylabel('Median Home Value')
plt.show()


# Elbow plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette plot
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Ignore all warnings.
warnings.filterwarnings('ignore')

# Load the dataset
boston_data = pd.read_csv("BostonHousing.csv")

# Display the first few rows of the dataset
print(boston_data.head())

# Check for missing values
print(boston_data.isnull().sum())

# Summary statistics
print(boston_data.describe())

# Histogram of crime rate (crim)
plt.figure(figsize=(8, 6))
plt.hist(boston_data['crim'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Crime Rate')
plt.xlabel('Crime Rate')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of average number of rooms (rm) vs. median value of owner-occupied homes (medv)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rm', y='medv', data=boston_data, color='purple')
plt.title('Average Number of Rooms vs. Median Home Value')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value')
plt.show()

# Create the heatmap for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(boston_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()



# Handle missing values (NaNs) by imputing the missing values with the mean of each column
from sklearn.impute import SimpleImputer
# Drop the target variable (assuming 'medv' is the target)
X = boston_data.drop(columns=['medv'])

# Imputer instance
imputer = SimpleImputer(strategy='mean')

# Impute missing values in the dataset
X_imputed = imputer.fit_transform(X)

# Data Preprocessing: Standardize features after imputation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Now continue with the rest of the code for clustering


# Determine the optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Fit k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
boston_data['Cluster'] = kmeans.labels_

# Visualize clustering results (scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rm', y='medv', hue='Cluster', data=boston_data, palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value')
plt.show()
# Histogram of cluster distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', data=boston_data)
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

# Bar chart of cluster distribution
plt.figure(figsize=(8, 6))
boston_data['Cluster'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Pie chart of cluster distribution
plt.figure(figsize=(8, 8))
boston_data['Cluster'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('Pastel1'))
plt.title('Distribution of Clusters')
plt.ylabel('')
plt.show()

# Line graph
plt.figure(figsize=(8, 6))
plt.plot(boston_data['age'], color='blue')
plt.title('Age Distribution')
plt.xlabel('Index')
plt.ylabel('Age')
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(boston_data['Cluster'], kmeans.labels_)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Heatmap (alternative way)
plt.figure(figsize=(10, 8))
sns.heatmap(boston_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot (Corner plot)
sns.pairplot(boston_data)
plt.title('Pairplot')
plt.show()

# Box plot
plt.figure(figsize=(10, 8))
sns.boxplot(x='Cluster', y='medv', data=boston_data)
plt.title('Box Plot of Cluster vs. Median Home Value')
plt.xlabel('Cluster')
plt.ylabel('Median Home Value')
plt.show()

# Violin plot
plt.figure(figsize=(10, 8))
sns.violinplot(x='Cluster', y='medv', data=boston_data)
plt.title('Violin Plot of Cluster vs. Median Home Value')
plt.xlabel('Cluster')
plt.ylabel('Median Home Value')
plt.show()


# Elbow plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Silhouette plot
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Ignore all warnings.
warnings.filterwarnings('ignore')

# Display the first few rows of the dataset
print(boston_data.head())

# Check for missing values
print(boston_data.isnull().sum())

# Summary statistics
print(boston_data.describe())

# Histogram of crime rate (crim)
def hist_crime_data(data):
    """
    Plot a histogram of crime rate data.

    Parameters:
    - data: DataFrame containing crime rate data.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data['crim'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Crime Rate')
    plt.xlabel('Crime Rate')
    plt.ylabel('Frequency')
    plt.show()

def scattoravg_of_room_and_home(a, b, boston_data):
    """
    Create a scatter plot of average number of rooms vs. median home value.

    Parameters:
    - a: Feature representing average number of rooms.
    - b: Feature representing median home value.
    - boston_data: DataFrame containing the dataset.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=a, y=b, data=boston_data, color='purple')
    plt.title('Average Number of Rooms vs. Median Home Value')
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Median Home Value')
    plt.show()

def heamap_corr(boston_data):
    """
    Create a heatmap of correlation between features in the dataset.

    Parameters:
    - boston_data: DataFrame containing the dataset.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(boston_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


# Handle missing values (NaNs) by imputing the missing values with the mean of each column
from sklearn.impute import SimpleImputer

# Imputer instance
imputer = SimpleImputer(strategy='mean')

# Impute missing values in the dataset
X_imputed = imputer.fit_transform(X)

# Data Preprocessing: Standardize features after imputation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Now continue with the rest of the code for clustering


# Determine the optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Fit k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
boston_data['Cluster'] = kmeans.labels_

plt.figure(figsize=(8, 6))
sns.scatterplot(x='rm', y='medv', hue='Cluster', data=boston_data, palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Median Home Value')
plt.show()

def line_index_age(data):
    """
    Create a line graph of age distribution.

    Parameters:
    - data: DataFrame containing the age data.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(data['age'], color='blue')
    plt.title('Age Distribution')
    plt.xlabel('Index')
    plt.ylabel('Age')
    plt.show()


plt.figure(figsize=(10, 8))
sns.boxplot(x='Cluster', y='medv', data=boston_data)
plt.title('Box Plot of Cluster vs. Median Home Value')
plt.xlabel('Cluster')
plt.ylabel('Median Home Value')
plt.show()

def elbow_curve():
    """
    Create an elbow plot to find the optimal number of clusters.

    Parameters:
    - None

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

# Silhouette plot
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)






#--------------------------------------------------------------------------------------------
# Load the dataset
boston_data = pd.read_csv("BostonHousing.csv")
hist_crime_data(boston_data)

x='rm'
y='medv'
scattoravg_of_room_and_home(x,y,boston_data)

heamap_corr(boston_data)


line_index_age(boston_data)


elbow_curve()


# In[ ]:




