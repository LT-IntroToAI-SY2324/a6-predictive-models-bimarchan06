import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles(000)", "age"]].values
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

# print out the linear equation and r^2 value
print(f"Model's Linear Equation: y={coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)

#Loop through the data and print out the predicted prices and the 
#actual prices
print("***************")
print("Testing Results")
# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)

#Testing the writeup questions:

carOne = model.predict([[89, 10]])
print("A 10 year old car with 89,000 miles is worth $", carOne)

carTwo = model.predict([[150, 20]])
print("A 20 year old car with 150,000 miles is worth $", carTwo)

carThree = model.predict([[500, 50]])
print("Car three is worth $", carThree)

# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)
print(predict)

# compare the actual and predicted values
print("\nTesting Multivariable Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index] # gets the actual y value from the ytest dataset
    predicted_y = predict[index] # gets the predicted y value from the predict variable
    x_coord = xtest[index] # gets the x value from the xtest dataset
    x_coord = np.around(x_coord, 2)
    print(f"Miles: {x_coord[0]} Age: {x_coord[1]} Actual: {actual} Predicted: {predicted_y}")