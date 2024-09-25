import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Generate synthetic data
np.random.seed(0)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Step 2: Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Step 3: Make predictions
X_new = np.array([[0], [2]])  # Make predictions for 0 and 2
y_predict = model.predict(X_new)

# Step 4: Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_predict, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

# Step 5: Output the coefficients
print(f"Intercept: {model.intercept_[0]}")
print(f"Slope: {model.coef_[0][0]}")

# v1 固定
# v2 可互動