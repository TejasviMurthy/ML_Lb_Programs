from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Replace these with your dataset
# Independent variable (X) and dependent variable (y)
X = np.array([[[2], [4], [6], [8], [10]]]).reshape(-1, 1)  # Example: [[1], [2], [3], [4], [5]]
y = np.array([0, 0, 1, 1, 1])  # Example: [0, 0, 1, 1, 1]

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities for visualization
x_values = np.linspace(min(X), max(X), 100).reshape(-1, 1)
y_prob = model.predict_proba(x_values)[:, 1]  # Probability of class 1

# Plotting the logistic regression curve
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(x_values, y_prob, color='red', label='Logistic Regression Curve')
plt.title('Simple Logistic Regression')
plt.xlabel('X')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

sample_value = [[5]]  # Replace with any value to test
predicted_class = model.predict(sample_value)
print(f"Predicted class for X = {sample_value}: {predicted_class}")
