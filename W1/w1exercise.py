import numpy as np
import pandas as pd
from array import array
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# a part(ii) normalize the data

def normalise(x):
    # calculate scaling factor
    scaling_factor = max(x)-min(x)
    # now normalise the data
    count = 0
    norm = np.array(x, dtype=float)
    for i in x:
        norm[count] = (i-min(x))/scaling_factor
        count = count+1
    return norm


def calsum(x, y, h_sum, mode):
    count = 0
    rt = 0
    tmp = 0
    while count < len(x):
        tmp = (h_sum[count] - y[count])
        if mode == 1:  # cal for theta1
            tmp = (tmp)*x[count]
        if mode == 2:  # cal for J(0)
            tmp = (tmp)**2
        rt = rt + tmp
        count = count+1
    return rt

# a part(iii) gradient descent


def gradient_descent(x, y, t0, t1, step_size):
    # create a gradient descent model.
    m = len(x)  # get m value
    cost = 100  # generic value for the cost function to start with
    cost_list = []  # list for graphing the costs
    i = 0  # index

    while i < 1000:
        # cal h(0) = t0 - t1(x(i)) for all elems of x
        h_theta = hsum(x, t0, t1)

        # get the new values for theta0 and theta1
        sumh0 = calsum(x, y, h_theta, 0)
        sumh1 = calsum(x, y, h_theta, 1)
        dif0 = (step_size*(-2))/m * sumh0
        dif1 = (step_size*(-2))/m * sumh1

        # cal new theta values
        t0 = t0 + dif0
        t1 = t1 + dif1

        # cal sum for cost funtion and add to the list
        h_theta = hsum(x, t0, t1)
        sumhth = calsum(x, y, h_theta, 2)
        cost = 1/m * (sumhth)
        cost_list.append(cost)
        i = i+1

    return t0, t1, cost, cost_list


def hsum(x, t0, t1):
    y_axis = []
    i = 0
    while i < len(x):
        y[i] = t0+(t1*x[i])
        i = i+1
    return y


# Q1 a (i)
# read in the data
df = pd.read_csv("week1.csv", comment='#')
x = np.array(df.iloc[:, 0], dtype=float)
x = x.reshape(-1, 1)
y = np.array(df.iloc[:, 1])
y = y.reshape(-1, 1)

# Q1 a(ii)
# Normalise the data
norm_x = normalise(x)
norm_y = normalise(y)

# Q1 a(iii)
theta0, theta1, cost, cost_list = gradient_descent(
    norm_x, norm_y, 0, 1, 0.1)  # call the function
plt.scatter(norm_x, norm_y, color='black', label='test data')
guess_plot_y = hsum(norm_x, theta0, theta1)
plt.plot(norm_x, guess_plot_y, color='Green', label='guess data')
plt.title(
    "Q1.a iii Training model vinalla Matthew Flynn 17327199")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.legend(["Model Vanilla", "Test data"])
plt.show()


# # Q1 b (i)
plt.plot(list(range(1000)), cost_list, '-b')  # plot the cost function.

theta0, theta1, cost, cost_list = gradient_descent(
    norm_x, norm_y, 0, 1, 0.001)  # call the function
plt.plot(list(range(1000)), cost_list, '-g')  # plot the cost function.

# Calculate the gradient descent and linear regression with different learning values

theta0, theta1, cost, cost_list = gradient_descent(
    norm_x, norm_y, 0, 1, 0.01)  # call the function
plt.plot(list(range(1000)), cost_list, '-r')  # plot the cost function.


theta0, theta1, cost, cost_list = gradient_descent(
    norm_x, norm_y, 0, 1, 0.1)  # call the function
plt.plot(list(range(1000)), cost_list, '-b')  # plot the cost function.

plt.title("Q1.b Cost with different learning rates Matthew Flynn 17327199")
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.legend(["alpha = 0.001", "alpha = 0.01", "alpha = 0.1", ])
plt.show()

# Q1 b(iii)
# cost of baseline vs trained
mean = sum(norm_x)/len(norm_x)
constant_output = [mean]
i = 0
while i < (len(norm_x)-1):
    constant_output.append(mean)
    i = i+1
const_cost = (1/len(constant_output)) * \
    calsum(norm_x, norm_y, constant_output, 2)
print(const_cost)
plt.plot(norm_x, constant_output, color='yellow', label='Const data')

# print scatter plot and vanilla for comparison
plt.scatter(norm_x, norm_y, color='black')
guess_plot_y = hsum(norm_x, theta0, theta1)
plt.plot(norm_x, guess_plot_y, color='Green', label='guess data')
plt.title(
    "Q1.b iii Comparing the Const model vers Vanilla model Matthew Flynn 17327199")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.legend(["constant output", "predictions vanilla"])
plt.show()


# Q1 b(iv)

# using sklearn
model = LinearRegression().fit(norm_x, norm_y)
y_pre = model.predict(norm_x)
sklearn_cost = (1/len(norm_x)) * calsum(norm_x, norm_y, y_pre, 2)
print("The Sklearn is = ", sklearn_cost)
guess_plot_y = hsum(norm_x, theta0, theta1)
print(model.intercept_, model.coef_)
plt.plot(norm_x, y_pre, color='red', label="sklearn")
plt.plot(norm_x, guess_plot_y, color='Green', label='guess data')
plt.scatter(norm_x, norm_y, color='black')
plt.title(
    "Q1.b iv Comparing the Sklearn model vers Vanilla model Matthew Flynn 17327199")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.legend(["predictions vanilla", "predictions sklearn", "training data"])
plt.show()
