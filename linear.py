import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 5 * np.random.rand(100, 1)  
y = 3 * X.squeeze() + np.random.randn(100) * 2 + 1  

plt.scatter(X, y, color='b', label='Data points')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Generated Data for Linear Regression')
plt.legend()
plt.show()

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Fitted Line')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

print(f'Linear Regression Coefficient: {model.coef_[0]:.2f}')
print(f'Linear Regression Intercept: {model.intercept_:.2f}')
