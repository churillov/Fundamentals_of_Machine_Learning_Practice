from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

boston = load_boston()
# print(boston["DESCR"])

x, y = boston["data"], boston["target"]

# plt.scatter(x[:, 0], y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn = KNeighborsRegressor(n_neighbors=5, weights="uniform", p=2)
knn.fit(x_train, y_train)

knn.predict(x_test)
# print(knn.predict(x_test))

mean_squared_error(y_test, knn.predict(x_test))
# print(mean_squared_error(y_test, knn.predict(x_test)))

grid_searcher = GridSearchCV(KNeighborsRegressor(),
                             param_grid={"n_neighbors": [1, 5, 10, 20],
                                         "weights": ["uniform", "distance"],
                                         "p": [1, 2, 3]},
                             cv=5)
grid_searcher.fit(x_train, y_train)
print(grid_searcher.best_params_)

mean_squared_error(y_test, grid_searcher.predict(x_test))
print(mean_squared_error(y_test, grid_searcher.predict(x_test)))


metrics = []
for n in range(1, 30, 3):
    knn = KNeighborsRegressor(n_neighbors=n)
    knn.fit(x_train, y_train)
    metrics.append(mean_squared_error(y_test, knn.predict(x_test)))

plt.plot(range(1, 30, 3), metrics)
plt.show()
