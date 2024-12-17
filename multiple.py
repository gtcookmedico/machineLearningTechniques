import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
X_multi = 5 * np.random.rand(100, 2)  
noise = np.random.randn(100) * 2
y_multi = 4 * X_multi[:, 0] + 2 * X_multi[:, 1] + 3 + noise  

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

print(f'Multiple Regression Coefficients: {model_multi.coef_}')
print(f'Multiple Regression Intercept: {model_multi.intercept_:.2f}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred_multi, color='red', label='Predicted', s=10)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Multiple Regression: Actual vs Predicted')
plt.legend()
plt.show()
