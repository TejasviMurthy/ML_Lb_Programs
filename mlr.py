import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data = {
    "feature1": [60, 62, 67, 70, 71, 72, 75, 78],
    "feature2": [22, 25, 24, 20, 15, 14, 11, 10],  # Ensure lengths match
    "target": [140, 155, 179, 192, 200, 212, 213, 210]  # Ensure lengths match
}
df = pd.DataFrame(data)

# Check if all columns have the same length
if len(df['feature1']) != len(df['feature2']) or len(df['feature1']) != len(df['target']):
    raise ValueError("All columns must be of the same length")

# Features and target variable
X = df[["feature1", "feature2"]].values
y = df["target"].values

# Add a bias (intercept) term to the feature matrix
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones to X

try:
    # Normal Equation: theta = (X_b.T * X_b)^(-1) * X_b.T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Making predictions
    y_pred = X_b.dot(theta_best)

    # Evaluation
    mse = np.mean((y - y_pred) ** 2)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    print("Coefficients (theta):", theta_best[1:])
    print("Intercept (bias):", theta_best[0])
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)


    fig = plt.figure(figsize=(10, 6))


    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y, color="b", label="Original Data")


    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 10)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)


    y_mesh = theta_best[0] + theta_best[1] * x1_mesh + theta_best[2] * x2_mesh


    ax.plot_surface(
        x1_mesh, x2_mesh, y_mesh, color="r", alpha=0.5
    )
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title("Multiple Linear Regression")
    ax.legend()

    plt.show()

except np.linalg.LinAlgError as e:
    print("Linear algebra error:", e)
except ValueError as e:
    print("Value error:", e)