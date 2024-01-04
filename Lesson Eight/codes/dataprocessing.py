# Example 1: Reading and Cleaning Data
import pandas as pd

# Read a CSV file into a DataFrame
df = pd.read_csv('example_data.csv')

# Display the first few rows of the DataFrame
print("Original Data:")
print(df.head())

# Data Cleaning: Remove missing values
df_cleaned = df.dropna()

# Display the cleaned DataFrame
print("\nCleaned Data:")
print(df_cleaned.head())



# Example 2: Data Transformation
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'Salary': [50000, 60000, 45000]}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original Data:")
print(df)

# Data Transformation: Add a new column
df['Salary_Adjusted'] = df['Salary'] * 1.1

# Display the transformed DataFrame
print("\nTransformed Data:")
print(df)


# Example 3: Data Analysis
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 22],
        'Salary': [50000, 60000, 45000]}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original Data:")
print(df)

# Data Analysis: Calculate average age
average_age = df['Age'].mean()

# Display the result
print(f"\nAverage Age: {average_age}")



# Example 4: Data Visualization
import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame
data = {'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'Population': [8398748, 3990456, 2705994, 2325502, 1680992]}
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original Data:")
print(df)

# Data Visualization: Create a bar chart
df.plot(kind='bar', x='City', y='Population', legend=False)
plt.title('Population of Cities')
plt.xlabel('City')
plt.ylabel('Population')
plt.show()


# Example 5: Data Processing Pipeline
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read a CSV file into a DataFrame
df = pd.read_csv('example_data.csv')

# Data Processing Pipeline
# 1. Remove missing values
df_cleaned = df.dropna()

# 2. Split the data into features and target
X = df_cleaned.drop('TargetColumn', axis=1)
y = df_cleaned['TargetColumn']

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Display the processed data
print("Processed Data:")
print(X_train_scaled[:3])

