import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = 5 * np.random.rand(100, 1).squeeze()
y = 3 * X + np.random.randn(100) * 2 + 1

def gradient_descent(X, y, lr=0.01, epochs=1000):
    m = len(y)
    theta_0, theta_1 = 0, 0
    cost_history = []

    for epoch in range(epochs):
        y_pred = theta_0 + theta_1 * X  
        error = y_pred - y
        cost = (1 / (2 * m)) * np.sum(error ** 2)  
        cost_history.append(cost)

        grad_theta_0 = (1 / m) * np.sum(error)
        grad_theta_1 = (1 / m) * np.sum(error * X)

        theta_0 -= lr * grad_theta_0
        theta_1 -= lr * grad_theta_1

    return theta_0, theta_1, cost_history

theta_0, theta_1, cost_history = gradient_descent(X, y, lr=0.01, epochs=1000)
print(f'Gradient Descent Results: Intercept = {theta_0:.2f}, Coefficient = {theta_1:.2f}')

plt.plot(range(1, 1001), cost_history)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost History During Gradient Descent')
plt.show()

y_gd_pred = theta_0 + theta_1 * X
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_gd_pred, color='green', label='Gradient Descent Fit')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression Fit using Gradient Descent')
plt.legend()
plt.show()
