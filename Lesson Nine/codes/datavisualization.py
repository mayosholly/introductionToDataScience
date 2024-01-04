# Example 1: Line Plot

import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line plot
plt.plot(x, y, label='Line Plot')

# Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# Example 2: Scatter Plot
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
x = np.random.rand(50)
y = 2 * x + 1 + 0.1 * np.random.randn(50)

# Create a scatter plot
plt.scatter(x, y, label='Scatter Plot', color='red', marker='o')

# Customize the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Regression Line')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# Example 3: Bar Chart
import matplotlib.pyplot as plt

# Sample data
categories = ['Category A', 'Category B', 'Category C']
values = [30, 50, 20]

# Create a bar chart
plt.bar(categories, values, color=['blue', 'green', 'orange'])

# Customize the plot
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')

# Show the plot
plt.show()


# Example 4: Histogram
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
data = np.random.randn(1000)

# Create a histogram
plt.hist(data, bins=20, color='skyblue', edgecolor='black')

# Customize the plot
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')

# Show the plot
plt.show()


# Example 5: Pie Chart
import matplotlib.pyplot as plt

# Sample data
labels = ['Category A', 'Category B', 'Category C']
sizes = [40, 30, 30]

# Create a pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['gold', 'lightcoral', 'lightskyblue'])

# Customize the plot
plt.title('Pie Chart Example')

# Show the plot
plt.show()

