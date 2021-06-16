import numpy as np
import matplotlib.pyplot as plt


N = 1000
D = 2

r_inner = 5
r_outer = 10

r1 = np.random.randn(N//2) + r_inner
theta = 2 * np.pi * np.random.randn(N//2)
X_inner = np.concatenate([[r1 * np.cos(theta)], [r1 * np.sin(theta)]]).T

r2 = np.random.randn(N//2) + r_outer
theta = 2 * np.pi * np.random.randn(N//2)
X_outer = np.concatenate([[r2 * np.cos(theta)], [r2 * np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
T = np.array([0]*(N//2) + [1]*(N//2))


plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

ones = np.ones((N, 1))
r = np.sqrt( (X * X).sum(axis=1) ).reshape(-1, 1)
Xb = np.concatenate((ones, r, X), axis=1)
w = np.random.randn(D + 2)
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()


learning_rate = 0.0001
error = []
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 500 == 0:
        print(e)

    w += learning_rate * (Xb.T.dot(T - Y) - 0.1 * w)
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.show()

print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)
