# Importing Pandas:

import pandas as pd


# Creating a DataFrame:

data = {'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 28, 22],
        'City': ['New York', 'San Francisco', 'Seattle']}

df = pd.DataFrame(data)


# Viewing Data:

# Display the first few rows
print(df.head())

# Display basic statistics
print(df.describe())


# Selecting Columns:

# Selecting a single column
age_column = df['Age']

# Selecting multiple columns
subset = df[['Name', 'City']]


# Filtering Data:

# Filtering based on a condition
young_people = df[df['Age'] < 30]


# Adding a New Column:

# Adding a new column
df['IsAdult'] = df['Age'] >= 18


# Reading Data from a File:

# Reading a CSV file into a DataFrame
file_path = 'path/to/your/file.csv'
df = pd.read_csv(file_path)



# Example of Reading Data and Converting to DataFrame:
# Name,Age,Salary
# John,25,50000
# Alice,30,60000
# Bob,28,55000


# To read this data into a Pandas DataFrame:
import pandas as pd

# Read CSV file into DataFrame
df = pd.read_csv('example.csv')

# Display the DataFrame
print(df)



# Reading Data with Pandas:

# CSV (Comma-Separated Values):

import pandas as pd

# Reading data from a CSV file
df_csv = pd.read_csv('your_file.csv')

# Display the DataFrame
print(df_csv)


# Excel:

import pandas as pd

# Reading data from an Excel file
df_excel = pd.read_excel('your_file.xlsx', sheet_name='Sheet1')

# Display the DataFrame
print(df_excel)


# SQL Database:

import pandas as pd
from sqlalchemy import create_engine

# Creating a SQL connection
engine = create_engine('sqlite:///your_database.db')

# Reading data from a SQL table
query = 'SELECT * FROM your_table'
df_sql = pd.read_sql(query, engine)

# Display the DataFrame
print(df_sql)



# JSON:

import pandas as pd

# Reading data from a JSON file
df_json = pd.read_json('your_file.json')

# Display the DataFrame
print(df_json)
Requirements in the Real World:

# Pandas Installation:

# Make sure Pandas is installed. You can install it using:
# bash
# Copy code
# pip install pandas
# Basic Python Knowledge:

# Familiarity with basic Python syntax is essential for using Pandas effectively.
# Understanding Data Structures:

# Learn about Pandas data structures, especially DataFrame and Series.
# Data Cleaning Skills:

# Pandas is often used for data cleaning, so understanding methods like handling missing values and duplicates is crucial.
# Exploratory Data Analysis (EDA):

# Use Pandas for EDA tasks such as grouping, aggregation, and visualization.
# Integration with Other Libraries:

# Learn how to integrate Pandas with other libraries like NumPy, Matplotlib, and Seaborn for comprehensive data analysis and visualization.
# SQL Knowledge (Optional):

# Understanding basic SQL can be helpful when working with Pandas for database-related tasks.
# Real-world Application:

# Apply Pandas to real-world datasets and problems to gain practical experience.
# Pandas is an invaluable tool in the data science toolkit, enabling efficient data manipulation and analysis. Learning Pandas will significantly enhance your ability to work with structured data and perform various data-related tasks in the real world.


