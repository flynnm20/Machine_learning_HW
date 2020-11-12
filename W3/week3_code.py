import numpy as np
import pandas as pd
from array import array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model 
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error 


def plot3d(Xtest,Ytest,x,y,legend):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    ax.scatter(x[:,0],x[:,1],y, color='black')
    ax.scatter(Xtest[:,0],Xtest[:,1],Ytest, color='green', marker='.')
    ax.legend(legend)
    ax.set_title("Matthew Flynn 17327199, Part i C")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target")


# read in the data taken from week 2
# (i) a, Read in data and display it in a 3d diagram
df = pd.read_csv("week3.csv", comment ="#")
x1 = df.iloc[:, 0]
x2 = df.iloc[:, 1]
x = np.column_stack((x1, x2))
y = df.iloc[:, 2]

#plot 3d figure. taken from his code
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.scatter(x[:,0],x[:,1],y)
ax.set_title("Matthew Flynn 17327199, Part i A")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Target")
plt.show()

# (i) b add polynomial features up to power of 5 
# then train a lasso regression model for C = 1, 10 ,1000
Xpoly = PolynomialFeatures(5).fit_transform(x)
C = [1,10, 100,1000]
for i in C:
    #print()
    clf = linear_model.Lasso(alpha = (1/(2*i)))
    lasso_model = clf.fit(Xpoly, y)
    print("coef = ", lasso_model.coef_)
    print("intercept = ", lasso_model.intercept_)

# (i) c predictions vs test data.
Xtest = []
Xpoly = PolynomialFeatures(5).fit_transform(x)
fig = plt.figure()
grid = np.linspace(-2 ,2)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
XtestPoly = PolynomialFeatures(5).fit_transform(Xtest)
C = [1,10, 100, 1000]
legend = ['test Data', 'Lasso model']
for i in C:
    clf = linear_model.Lasso(alpha = (1/(2*i)))
    lasso_model = clf.fit(Xpoly, y)
    Ytest = clf.predict(XtestPoly)
    plot3d(Xtest,Ytest,x,y,legend)
    plt.show()

# (e)
#then train a Ridge regression model for C = 1, 10 ,1000
Xpoly = PolynomialFeatures(5).fit_transform(x)
C = [1,10, 100,1000]
for i in C:
    clf = linear_model.Ridge(alpha = (1/(2*i)))
    ridge_model = clf.fit(Xpoly, y)
    print("coef = ", ridge_model.coef_)
    print("intercept = ", ridge_model.intercept_)

Xtest = []
Xpoly = PolynomialFeatures(5).fit_transform(x)
grid = np.linspace(-2 ,2)
for i in grid:
    for j in grid:
        Xtest.append([i,j])
Xtest = np.array(Xtest)
XtestPoly = PolynomialFeatures(5).fit_transform(Xtest)
C = [1,10, 100, 1000]
legend = ['test Data', 'Ridge model']
for i in C:
    fig = plt.figure()
    clf = linear_model.Ridge(alpha = (1/2*i))
    ridge_model = clf.fit(Xpoly, y)
    Ytest = clf.predict(XtestPoly)
    ax = fig.add_subplot(111, projection ='3d')
    ax.scatter(x[:,0],x[:,1],y, color='black')
    ax.scatter(Xtest[:,0],Xtest[:,1],Ytest, color='green', marker='.')
    ax.legend(legend)
    ax.set_title("Matthew Flynn 17327199, Part i e")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Target")
    plt.show()



# ii (a)
c_start = 1
x_Poly = PolynomialFeatures(5).fit_transform(x)
fold = [5,10,25,50,100]
for i in fold:
    k_f = KFold(n_splits=i)
    model = linear_model.Lasso(alpha=(1/(2*c_start)))
    for trainer, test in k_f.split(x_Poly):
        model.fit(x_Poly[trainer], y[trainer])
    scores = cross_val_score(model, x_Poly, y, cv=i, scoring="neg_mean_squared_error")
    print("mean = ", scores.mean(), "+/- ", scores.std())

# ii graphing b
x_Poly = PolynomialFeatures(5).fit_transform(x)
mean_error = []
std_error =[]
tmp =[]
C = [1,10, 100,1000]
k_f = KFold(n_splits=5)
for i in C:
    model = linear_model.Lasso(alpha=1/(2*i))
    for trainer , tester in k_f.split(x_Poly):
        model.fit(x_Poly[trainer], y[trainer])
        Ytest = model.predict(x_Poly[tester])
        tmp.append(mean_squared_error(y[tester],Ytest))
    mean_error.append(np.array(tmp).mean())
    std_error.append(np.array(tmp).std())
plt.rc('font',size = 18)
plt.title("Matthew Flynn 17327199 part 2 b")
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(C , mean_error ,yerr=std_error, linewidth=4)
plt.xlabel("C")
plt.ylabel("mean square error")
plt.show()

# ii (d)
mean_error = []
std_error =[]
tmp =[]
C = [1,10, 100,1000]
x_Poly = PolynomialFeatures(5).fit_transform(x)
for i in C:
    k_f = KFold(n_splits=10)
    model = Ridge(alpha=1/(2*i))
    for trainer , tester in k_f.split(x_Poly):
        model.fit(x_Poly[trainer], y[trainer])
        Ytest = model.predict(x_Poly[tester])
        tmp.append(mean_squared_error(y[tester],Ytest))
    mean_error.append(np.array(tmp).mean())
    std_error.append(np.array(tmp).std())
plt.errorbar(C , mean_error ,yerr=std_error, linewidth=4)
plt.title("Matthew Flynn 17327199 part 2 d")
plt.xlabel("C")
plt.ylabel("mean square error")
plt.show()