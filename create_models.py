import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression

x = np.array([[1], [2], [3]])
y = np.array([[1], [3], [6]])
linear = LinearRegression().fit(x, y)

result = linear.predict([[3.0]])

print("Linear Prediction for 3.0 : ", result[0])

joblib.dump(linear, "linear.joblib")

x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([0, 0, 1, 1])
logistic = LogisticRegression().fit(x, y)

result = logistic.predict([[2, 1], [3, 3]])

print("Logistic Prediction for [2, 1] : ", result[0])

joblib.dump(logistic, "logistic.joblib")
