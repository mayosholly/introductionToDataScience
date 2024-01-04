# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets  # Import datasets module
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random data for regression
X_reg = 2 * np.random.rand(100, 1)
y_reg = 4 + 3 * X_reg + np.random.randn(100, 1)

# Generate random data for classification
X_cls, y_cls = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split data for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Split data for classification
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Linear Regression for Regression
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# SVM for Regression
svm_reg_model = SVR(kernel='linear')
svm_reg_model.fit(X_reg_train, y_reg_train.ravel())
y_svm_reg_pred = svm_reg_model.predict(X_reg_test)

# Random Forest for Regression
rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg_model.fit(X_reg_train, y_reg_train.ravel())
y_rf_reg_pred = rf_reg_model.predict(X_reg_test)

# SVM for Classification
svm_cls_model = SVC(kernel='linear')
svm_cls_model.fit(X_cls_train, y_cls_train)
y_svm_cls_pred = svm_cls_model.predict(X_cls_test)

# Random Forest for Classification
rf_cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls_model.fit(X_cls_train, y_cls_train)
y_rf_cls_pred = rf_cls_model.predict(X_cls_test)

# Evaluate models
mse_reg = mean_squared_error(y_reg_test, y_reg_pred)
mse_svm_reg = mean_squared_error(y_reg_test, y_svm_reg_pred)
mse_rf_reg = mean_squared_error(y_reg_test, y_rf_reg_pred)

accuracy_svm_cls = accuracy_score(y_cls_test, y_svm_cls_pred)
accuracy_rf_cls = accuracy_score(y_cls_test, y_rf_cls_pred)

# Display results
print(f'Mean Squared Error (Linear Regression): {mse_reg}')
print(f'Mean Squared Error (SVM for Regression): {mse_svm_reg}')
print(f'Mean Squared Error (Random Forest for Regression): {mse_rf_reg}')
print(f'Accuracy (SVM for Classification): {accuracy_svm_cls}')
print(f'Accuracy (Random Forest for Classification): {accuracy_rf_cls}')



Certainly! Let's generate random data with a specified seed using NumPy and then apply various Scikit-learn models for prediction. We'll explore Linear Regression, Support Vector Machine (SVM), and Random Forest for both regression and classification tasks.

python
Copy code
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random data for regression
X_reg = 2 * np.random.rand(100, 1)
y_reg = 4 + 3 * X_reg + np.random.randn(100, 1)

# Generate random data for classification
X_cls, y_cls = datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split data for regression
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Split data for classification
X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Linear Regression for Regression
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# SVM for Regression
svm_reg_model = SVR(kernel='linear')
svm_reg_model.fit(X_reg_train, y_reg_train.ravel())
y_svm_reg_pred = svm_reg_model.predict(X_reg_test)

# Random Forest for Regression
rf_reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg_model.fit(X_reg_train, y_reg_train.ravel())
y_rf_reg_pred = rf_reg_model.predict(X_reg_test)

# SVM for Classification
svm_cls_model = SVC(kernel='linear')
svm_cls_model.fit(X_cls_train, y_cls_train)
y_svm_cls_pred = svm_cls_model.predict(X_cls_test)

# Random Forest for Classification
rf_cls_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cls_model.fit(X_cls_train, y_cls_train)
y_rf_cls_pred = rf_cls_model.predict(X_cls_test)

# Evaluate models
mse_reg = mean_squared_error(y_reg_test, y_reg_pred)
mse_svm_reg = mean_squared_error(y_reg_test, y_svm_reg_pred)
mse_rf_reg = mean_squared_error(y_reg_test, y_rf_reg_pred)

accuracy_svm_cls = accuracy_score(y_cls_test, y_svm_cls_pred)
accuracy_rf_cls = accuracy_score(y_cls_test, y_rf_cls_pred)

# Display results
print(f'Mean Squared Error (Linear Regression): {mse_reg}')
print(f'Mean Squared Error (SVM for Regression): {mse_svm_reg}')
print(f'Mean Squared Error (Random Forest for Regression): {mse_rf_reg}')
print(f'Accuracy (SVM for Classification): {accuracy_svm_cls}')
print(f'Accuracy (Random Forest for Classification): {accuracy_rf_cls}')


# Code Explanation:
# Random Seed:

# np.random.seed(42) sets the random seed for reproducibility.
# Generate Random Data:
# np.random.seed(42) is a command that ensures the reproducibility of random processes.

# Generates random data for regression and classification tasks.
# Split Data:

# Splits the data into training and testing sets for both regression and classification.
# Linear Regression for Regression:

# Trains a Linear Regression model and makes predictions.
# SVM for Regression:

# Trains an SVM model for regression using a linear kernel and makes predictions.
# Random Forest for Regression:

# Trains a Random Forest model for regression and makes predictions.
# SVM for Classification:

# Trains an SVM model for classification using a linear kernel and makes predictions.
# Random Forest for Classification:

# Trains a Random Forest model for classification and makes predictions.
# Evaluate Models:

# Calculates mean squared error for regression models and accuracy for classification models.