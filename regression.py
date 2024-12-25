import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

data = {
    'x': [10, 12, 16, 11, 15, 14, 20, 22],
    'y': [15, 18, 23, 14, 20, 17, 25, 28]
}


df = pd.DataFrame(data)


df['x^2'] = df['x'] ** 2
df['y^2'] = df['y'] ** 2
df['x*y'] = df['x'] * df['y']


df.to_excel('data.xlsx', index=False)


data = pd.read_excel('data.xlsx')

x = data['x'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)


regression_x_on_y = LinearRegression()
regression_x_on_y.fit(y, x)
slope_x_on_y = regression_x_on_y.coef_[0][0]
intercept_x_on_y = regression_x_on_y.intercept_[0]


regression_y_on_x = LinearRegression()
regression_y_on_x.fit(x, y)
slope_y_on_x = regression_y_on_x.coef_[0][0]
intercept_y_on_x = regression_y_on_x.intercept_[0]


print(f"Regression equation of x on y: x = {slope_x_on_y:.4f} * y + {intercept_x_on_y:.4f}")
print(f"Regression equation of y on x: y = {slope_y_on_x:.4f} * x + {intercept_y_on_x:.4f}")


y_vals_for_x_on_y = np.linspace(min(y), max(y), 100)
x_vals_for_y_on_x = np.linspace(min(x), max(x), 100)


x_on_y_line = slope_x_on_y * y_vals_for_x_on_y + intercept_x_on_y
y_on_x_line = slope_y_on_x * x_vals_for_y_on_x + intercept_y_on_x


plt.scatter(x, y, color='blue', label='Data Points')


plt.plot(x_on_y_line, y_vals_for_x_on_y, color='red', label='x on y (x = a*y + b)')
plt.plot(x_vals_for_y_on_x, y_on_x_line, color='green', label='y on x (y = c*x + d)')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression Lines of x on y and y on x')
plt.legend()


plt.grid(True)
plt.savefig('regression_plot.png')  # Save the plot as an image
plt.show()  # Display the plot
