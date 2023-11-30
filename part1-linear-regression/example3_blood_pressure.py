import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
Run the program and consider the following questions:
1. Look at the data points on the graph. Do age and blood pressure appear to have a linear relationship?
2. What does the value of r tell you about the relationship between age and blood pressure?
'''

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"]
y = data["Blood Pressure"]

#sets the size of the graph

x = x.reshape(-1, 1)


plt.figure(figsize=(5,4))

model = LinearRegression().fit(x, y)

#labels axes and creates scatterplot
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.title("Systolic Blood Pressure by Age")
plt.scatter(x, y)

print("Pearson's Correlation: r = :", x.corr(y))
x = x.reshape(-1, 1)

# create the model
model = LinearRegression().fit(x, y)

# find the coefficient, bias, and r squared values
# each should be a float and rounded to two decimal places
coef = round(float(model.coef_), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# value you are going to predict
x_predict = 25
# plug that value into your model
prediction = model.predict([[x_predict]])

# print out the linear equation and r squared value
print(f"Model's Linear Equation: y = {coef}x + {intercept}")
print(f"R Squared value: {r_squared}")
print(f"Prediction when x is {x_predict}: {prediction}")

'''
The following code creates the graph to visualize the data
'''

# creates a scatter plot of originial data in purple
# and the predicted data in blue
plt.scatter(x,y, c="purple")
plt.scatter(x_predict, prediction, c="blue")


# plot the line of best fit in red and label the line
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# show the plot and legend
plt.legend()
plt.show()
