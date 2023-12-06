import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv("part3-multivariable-linear-regression/antelope_data.csv")
x_1 = data[""]
x_2 = data["Winter Severity"]
y = data["Fawn"]

fig, graph = plt.subplots(3)
graph[0].scatter(x_1, y)
graph[0].set_xlabel("Annual Precipitation")
graph[0].set_ylabel("Fawn")

graph[1].scatter(x_2, y)
graph[1].set_xlabel("Winter Severity")
graph[1].set_ylabel("Fawn")

graph[2].scatter(x_3, y)
graph[2].set_xlabel("Adult Population")
graph[2].set_ylabel("Fawn")

print("Correlation between Annual Precipitation and Fawn Population:",round(x_1.corr(y),2))
print("Correlation between Winter Severity and Fawn Population:",round(x_2.corr(y),2))
print("Correlation between Adult Population and Fawn Population:",round(x_3.corr(y),2))

plt.tight_layout()
plt.show()

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles","age"]].values
y = data["Price"].values


#split the data into training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)

#create linear regression model
model = LinearRegression().fit(xtrain, ytrain)

#Find and print the coefficients, intercept, and r squared values. 
#Each should be rounded to two decimal places. 
coef = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)

print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {coef[2]}x3 + {intercept}")
print("R Squared value:", r_squared)

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")