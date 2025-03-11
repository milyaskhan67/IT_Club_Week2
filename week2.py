import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Iris_Dataset.csv')

# Display summary statistics
print(data.describe())

# Histogram
data.hist(bins=20, figsize=(10, 6))
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.title('Sepal Length vs Sepal Width')
sns.scatterplot(
    data=data, 
    x='SepalLength', 
    y='SepalWidth', 
    s=100,
    hue='Species'
)
plt.show()

# Print Null values before cleaning
print("\nMissing Values before cleaning:")
print(data.isnull().sum())

# Handle missing values
data = data.dropna()

# Remove outliers using Z-score
z_score = stats.zscore(data.select_dtypes(include=[float, int]))
abs_z_scores = abs(z_score)
filter_entries = (abs_z_scores < 3).all(axis=1)
data = data[filter_entries]

# Scale numerical features
scaler = StandardScaler()
numerical_features = data.select_dtypes(include=[float, int]).columns
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Print cleaned and scaled data
print("\nData after scaling:")
print(data.head())
