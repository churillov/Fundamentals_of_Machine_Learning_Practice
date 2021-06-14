from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

x, y = make_moons(n_samples=1000, noise=0.25)
print(x.shape)

plt.scatter(x[:, 0], x[:, 1], c=y)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=5)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn.fit(x_train, y_train)

knn.predict(x_test)
print(knn.predict(x_test))

accuracy_score(y_test, knn.predict(x_test))
print(accuracy_score(y_test, knn.predict(x_test)))

x_space = np.linspace(-2, 2, 100)

x_grid, y_grid = np.meshgrid(x_space, x_space)
xy = np.stack([x_grid, y_grid], axis=2).reshape(-1, 2)

plt.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.3, c=knn.predict(xy))
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.show()