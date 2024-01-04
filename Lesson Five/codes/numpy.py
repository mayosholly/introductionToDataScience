# basic calculation using numpy without array
import numpy as np

# Basic Arithmetic
result_addition = np.add(5, 3)
result_subtraction = np.subtract(8, 2)
result_multiplication = np.multiply(4, 6)
result_division = np.divide(9, 3)

print("Addition Result:", result_addition)
print("Subtraction Result:", result_subtraction)
print("Multiplication Result:", result_multiplication)
print("Division Result:", result_division)

# Exponentiation
result_exponentiation = np.power(2, 3)
print("Exponentiation Result:", result_exponentiation)

# Square Root
result_square_root = np.sqrt(16)
print("Square Root Result:", result_square_root)

# Trigonometric Functions
angle_in_radians = np.pi / 4
result_sin = np.sin(angle_in_radians)
result_cos = np.cos(angle_in_radians)

print("Sine Result:", result_sin)
print("Cosine Result:", result_cos)

# Sum, Mean, and Standard Deviation
total_sum = np.sum([1, 2, 3, 4, 5])
mean_value = np.mean([10, 20, 30, 40, 50])
std_deviation = np.std([5, 10, 15, 20, 25])

print("Sum:", total_sum)
print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)


# 1. Addition:

import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

result_addition = arr1 + arr2
print("Addition Result:", result_addition)


# 2. Subtraction:

result_subtraction = arr2 - arr1
print("Subtraction Result:", result_subtraction)


# 3. Multiplication:

result_multiplication = arr1 * arr2
print("Multiplication Result:", result_multiplication)

# 4. Division:

result_division = arr2 / arr1
print("Division Result:", result_division)


# 5. Exponentiation:

arr3 = np.array([2, 3, 4])
result_exponentiation = np.power(arr3, 2)
print("Exponentiation Result:", result_exponentiation)



# 6. Square Root:

result_square_root = np.sqrt(arr3)
print("Square Root Result:", result_square_root)


# 7. Trigonometric Functions:

angle_in_radians = np.pi / 2
result_sin = np.sin(angle_in_radians)
result_cos = np.cos(angle_in_radians)

print("Sine Result:", result_sin)
print("Cosine Result:", result_cos)


# 8. Sum, Mean, and Standard Deviation:

arr4 = np.array([10, 20, 30, 40, 50])

total_sum = np.sum(arr4)
mean_value = np.mean(arr4)
std_deviation = np.std(arr4)

print("Sum:", total_sum)
print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)



# 9. Element-wise Comparison:

arr5 = np.array([1, 2, 3, 4, 5])
arr6 = np.array([5, 4, 3, 2, 1])

result_comparison = arr5 > arr6
print("Element-wise Comparison Result:", result_comparison)



# Examples of NumPy Usage:

import numpy as np

# Creating NumPy arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

# Performing vectorized operations
result = arr1 + arr2

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
scaled_matrix = matrix * 2

# Mathematical functions
sin_values = np.sin(np.array([0, np.pi/2, np.pi]))

# Linear algebra operations
matrix_product = np.dot(matrix, matrix.T)

# Random number generation
random_matrix = np.random.rand(3, 3)

# Indexing and slicing
first_row = matrix[0, :]
column_sum = np.sum(matrix, axis=0)
