import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

data = load_diabetes(as_frame=True)
y = data.target
data = data.frame
print(data)
x = data("bmi")
print(x)
print(y)

x = x.reshape(-1,1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

model = linear_model.LinearRegression().fit(xtrain, ytrain)

coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)

print(coef, intercept)

prediction = model.predict(xtest)

print(f"Coefficient: {coef}")
print(f"Mean squared error: {mean_squared_error(ytest, prediction)}")
print(f"Coefficient of determiniation: {r2_score(ytest, prediction)}")

# Plot the points 
plt.scatter(xtest, ytest, c="red")
plt.scatter(xtrain, ytrain, c="purple")
# plt.plot(xtrain, ytrain, c="blue", linewidth=3)
plt.plot(xtest, coef*xtest + intercept, c="r", label="Line of Best Fit")

plt.xlabel("bmi")
plt.ylabel("quantitative measure of disease progression one year after baseline")
plt.title("quantitative measure of disease progression one year after baseline by bmi")

plt.legend()