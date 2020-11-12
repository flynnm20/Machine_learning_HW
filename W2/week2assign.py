import numpy as np
import pandas as pd
from array import array
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# plot the data for part a (i)
def seperate_pos_neg(x1, x2, y):
    num_of_positives = 0
    index = 0
    pos_tmp_array = []
    pos_tmp_x_vals = []
    neg_tmp_x_vals = []
    neg_tmp_array = []
    index = 0
    while index < len(y):
        if y[index] == 1:
            pos_tmp_array.append(x2[index])
            pos_tmp_x_vals.append(x1[index])
        else:
            neg_tmp_array.append(x2[index])
            neg_tmp_x_vals.append(x1[index])
        index = index+1
    pos_vals = np.array(pos_tmp_array)
    neg_vals = np.array(neg_tmp_array)
    pos_x_vals = np.array(pos_tmp_x_vals)
    neg_x_vals = np.array(neg_tmp_x_vals)
    return pos_vals, neg_vals, pos_x_vals, neg_x_vals


def plot_descision_boundary(model):
    intercept = model.intercept_
    coef1, coef2 = model.coef_.T
    c = -intercept/coef2
    m = -coef1/coef2
    xd = np.array([min(x1), max(x1)])
    yd = m*xd + c
    plt.plot(xd, yd)


def plot_training_data_against_model(pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals, x, model, title, legend):
    y_predict = model.predict(x)
    pos_y_vals_pred, neg_y_vals_pred, pos_x_vals_pred, neg_x_vals_pred = seperate_pos_neg(
        x1, x2, y_predict)
    pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals = seperate_pos_neg(
        x1, x2, y)
    plt.scatter(pos_x_vals, pos_y_vals, color="green", marker="o")
    plt.scatter(neg_x_vals, neg_y_vals, color="blue", marker="+")
    plt.scatter(pos_x_vals_pred, pos_y_vals_pred, color="red", marker="x")
    plt.scatter(neg_x_vals_pred, neg_y_vals_pred, color="black", marker=".")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(legend)


def new_features(x1, x2):
    x3 = x1**2
    x4 = x2**2
    return x3, x4


def baselineModel(pos_x_vals, neg_x_vals, input_data_1, input_data_2):
    if len(pos_x_vals) > len(neg_x_vals):
        return input_data_1, input_data_2, [], []
    else:
        return [], [], input_data_1, input_data_2


# read in the data
df = pd.read_csv("week2.csv")
x1 = np.array(df.iloc[:, 0])
x2 = np.array(df.iloc[:, 1])
x = np.column_stack((x1, x2))
y = df.iloc[:, 2]

# a (i) visualise the data
pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals = seperate_pos_neg(x1, x2, y)
plt.scatter(pos_x_vals, pos_y_vals, color="green", marker="x")
plt.scatter(neg_x_vals, neg_y_vals, color="red", marker="o")
plt.title("Week 2 A (i) Matthew Flynn 17327199")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend(['positive test values', 'negative test values'])
plt.show()

# a (ii)
model = LogisticRegression(penalty="none", solver="lbfgs")
model.fit(x, y)
print("intercept", model.intercept_, "  coeff ", model.coef_)

# a (iii)
# get intercept and slope for the decision boundary
plot_descision_boundary(model)
plot_training_data_against_model(
    pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals, x, model, "Week 2 A (iii) Matthew Flynn 17327199", ['decision boundary', 'test data positive', 'test data negative',
                                                                                                        'Model Positive', 'Model Negative'])
plt.show()


# b (i)
# train SVM classifiers over a wide range of Cs
classifiers_List = np.array([0.001, 0.01, 0.1, 1, 10])
trained_Classifiers = []
for i in classifiers_List:
    model = LinearSVC(C=i).fit(x, y)
    print("C = ", i, "intercept = ", model.intercept_,
          "Coefficient = ", model.coef_)
    trained_Classifiers.append(model)

# b (ii)
# trained classifiers maintains the data.
for i in trained_Classifiers:
    y_predict = i.predict(x)
    pos_y_vals_pred, neg_y_vals_pred, pos_x_vals_pred, neg_x_vals_pred = seperate_pos_neg(
        x1, x2, y_predict)
    plot_descision_boundary(i)
    plot_training_data_against_model(pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals, x, i, "Week 2 B (ii) Matthew Flynn 17327199", ['decision boundary', 'test data positive', 'test data negative',
                                                                                                                                    'Model Positive', 'Model Negative'])
    plt.show()


# C (i)
# new features are x1 and x2 squared
x3, x4 = new_features(x1, x2)
new_features = np.column_stack((x1, x2, x3, x4))
model = LogisticRegression(penalty="none", solver="lbfgs")
model.fit(new_features, y)
print("intercept", model.intercept_, "  coeff ", model.coef_)

# C (ii)
plot_training_data_against_model(
    pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals, new_features, model, "Week 2 C (ii) Matthew Flynn 17327199", ['test data positive', 'test data negative',
                                                                                                                  'Model Positive', 'Model Negative'])
plt.show()

#C (iii)
base_x_pos, base_y_pos, base_x_neg, base_y_neg = baselineModel(
    pos_x_vals, neg_x_vals, x1, x2)
# model that predicts the same result for everything.
plt.scatter(base_x_pos, base_y_pos, color="red", marker='+')
plt.scatter(base_x_neg, base_y_neg, color="yellow", marker='o')
plt.title("Week 2 C (iii) Matthew Flynn 17327199")
pos_y_vals, neg_y_vals, pos_x_vals, neg_x_vals = seperate_pos_neg(
    x1, x2, y)
plt.scatter(pos_x_vals, pos_y_vals, color="green", marker=".")
plt.scatter(neg_x_vals, neg_y_vals, color="blue", marker="+")
plt.legend(['baseline Model positive', 'baseline Model negative',
            'test data positive', 'test data negative'])
plt.show()
