import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use("ggplot")

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

#read the data set
data = pd.read_csv("data.csv")
#separate the grades and results
grades = data[['grade1' , 'grade2']].values
results = data[['result']].values
#visulize the data

passed = (results == 1).reshape(100, 1)
failed =  (results == 0).reshape(100, 1)
#separation of faile points and passe points

ax=sns.scatterplot(x= grades[passed[:,0], 0],
                y= grades[passed[:,0], 1],
                marker = "^",
                color = "blue",
                s = 60 )

sns.scatterplot(x= grades[failed[:,0], 0],
                y= grades[failed[:,0], 1],
                marker = "x",
                color = "red",
                s = 60)
ax.set(xlabel= "test 1 result", ylabel= 'test 2 result')

ax.legend(['passed', 'failed'])
#plt.show()

#definong the sigmoid function
def sigmoid_function(x):
    return 1/(1+ np.exp(-x))
#Copute the cost funtcion and gradient
'''the objective of the logistic regression is to minimize the cost function'''
def compute_cost(theta, x, y):
    m = len(y)
    y_predict = sigmoid_function(np.dot(x,theta))
    error = (y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict ))
    cost = 1/m * sum(error)
    gradient = 1/m * np.dot(x.transpose() , (y_predict - y))
    return cost , gradient
# Cost And Gradient at Initialization
mean_grades = np.mean(grades, axis=0)
std_grades = np.std(grades , axis=0)

rows = grades.shape[0]
cols = grades.shape[1]

X = np.append(np.ones((rows , 1)), grades , axis=1)
y = results.reshape(rows, 1)

theta_init = np.zeros((cols + 1, 1))
cost, gradient = compute_cost(theta_init, X , y)
#print( gradient , cost)

#Gradient Descent
'''Minimize the cost function by updating the gradient descent equation and repeat until convergence'''
def gradient_descent(x, y, theta, alpha, iteration):
    costs = []
    for i in range(iteration):
        cost, gradient = compute_cost(theta, x, y)
        theta -= (alpha * gradient)
        costs.append(cost)
    return theta, costs

theta , costs = gradient_descent(X, y, theta_init,1 , 200)
#print("theta after 200 iteration", theta)
#print(costs[-1])

#Plotting the convergence of compute cost
'''plot compute cost against the number of iterations of gradient descent'''
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("values of cost function over Iteration of gradient descent")

#Plotting the decison boundry
ax=sns.scatterplot(x= X[passed[:,0], 1],
                y= X[passed[:,0], 2],
                marker = "^",
                color = "blue",
                s = 60 )

sns.scatterplot(x= X[failed[:,0], 1],
                y= X[failed[:,0], 2],
                marker = "x",
                color = "red",
                s = 60 )
ax.legend(['passed', 'failed'])
ax.set(xlabel="test 1 grades" , ylabel="test 2 grades")

x_boundary = np.array([np.min(X[:, 1]), np.max[:, 1]])
y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]

sns.lineplot(x=x_boundary, y=y_boundary, color = "green")
plt.show()


#Predictions using the optimezed theta values
def predict(theta, x):
    results = x.dot(theta)
    return results > 0

p = predict(theta, X)
print("training accuracy:", sum(p==y)[0], "%")

test = np.array([50,70])
test = (test - mean_grades)/std_grades
test = np.append(np.ones(1), test)
probability = sigmoid_function(test.dor(theta))
print("person who grade 50 and 70 on their test have",
      np.round(probability[0], 3), "probability of passing the test")

