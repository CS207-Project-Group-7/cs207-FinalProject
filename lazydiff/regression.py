from lazydiff.vars import Var
from lazydiff import ops
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def MSE(X, y, m, b):
    """
    Returns Mean Squared Error where
    the predicted target is mX+b
    and y is the observed target variable
    """
    loss = Var(0)
    for vec, y_i in zip(X,y):
        loss = loss + (ops.sum(m*vec)+b-y_i)**2
    return loss/len(X)

def MSE_regularized(X, y, m, b, p = 1, C = 1):
    """
    Returns a regularized Mean Squared Error with L-p norm
    where the predicted target is mX+b,
    y is the observed target variable
    and C is the weight in L-p norm of the vector m
    """
    loss = Var(0)
    for vec,y_i in zip(X,y):
        loss = loss + (ops.sum(m*vec)+b-y_i)**2
    return loss/(2*len(X)) + C*ops.norm(m, p=p)**p

def lasso_loss(X, y, m, b, C = 1):
    """
    Returns Ridge Regression objective function
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    the predicted target is mX+b,
    y is the observed target variable
    and C is the weight in L-1 norm of the vector m
    """
    return MSE_regularized(X, y, m, b, p = 1, C = C)

def ridge_loss(X, y, m, b, C = 1):
    """
    Returns Ridge Regression objective function
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
    the predicted target is mX+b,
    y is the observed target variable
    and C is the weight in L-2 norm of the vector m
    """
    loss = Var(0)
    for vec, y_i in zip(X,y):
        loss = loss + (ops.sum(m*vec)+b-y_i)**2
    return loss + C*ops.norm(m,2)**2

def elastic_loss(X, y, m, b, C = 1, l1_ratio = 0.5):
    """
    Returns Elastic Net Regression objective function
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
    the predicted target is mX+b,
    y is the observed target variable,
    C is the weight in L-2 norm of the vector m
    and L1_ratio is the ratio of L-1 norm loss
    """
    loss = Var(0)
    for vec,y_i in zip(X,y):
        loss = loss + (ops.sum(m*vec)+b-y_i)**2
    return loss/(2*len(X)) + C*l1_ratio*ops.norm(m, p=1) + 0.5*C*(1-l1_ratio)*ops.norm(m,2)**2

def gradient_descent(X, y, loss_function, m, b, lr = 0.1, forward = True):
    """ Performs one single update step of gradient descent
        Returns the updated parameters m, b and loss
        X is the matrix of independent variables
        y is the vector of dependent variable
        m is the coefficient of the prediction mX+b
        b is the intercept/bias of the prediction
        lr is the learning rate
        forward determines whether to perform forward mode
        or reverse mode to find the gradient 
    """
    loss = loss_function(X, y, m, b)
    if (forward):
        # forward mode
        m.forward()
        b.forward()
    else:
        # reverse mode
        loss.backward()
    # clear cache by reinstantiating
    m = Var(m.val-lr*loss.grad(m))
    b = Var(b.val-lr*loss.grad(b))
    return m, b, loss

def timer(f):
    """
    timer decorator from CS207 lecture 4
    """
    def inner(*args):
        t0 = time.time()
        output = f(*args)
        elapsed = time.time() - t0
        print("Time Elapsed", elapsed)
        return output, elapsed
    return inner

@timer
def iterative_regression(X, y, m, b, loss_function, lr = 0.1,\
        epochs = 100, earlyStop = 0, forward = True, plot = False):
    """
    Performs iterative regression with the given loss function
    minimizing the loss function w.r.t. the parameters

    Returns the updated parameters m, b and the minimized loss
    X, y are the dependent and independent variables
    m, b are the coefficients and intercept
    loss_function is the objective function
    lr is the learning rate of the gradient descent
    epochs is the number of updates over the entire training data set
    earlyStop is absolute tolerance to stop the iteration early
    Note that 0 means no early stopping
    forward determines whether to perform forward mode
    or reverse mode to find the gradient 
    plot determines whether or not to plot
    """
    
    # cannot plot unless dimension of X is 1
    canPlot = X.shape[1] == 1 and plot
    
    if (canPlot):
        # plot the iterative regression 
        # and the linear regression from sklearn as reference
        plt.figure()
        plt_range = np.concatenate([min(X),max(X)])
        fig,ax = plt.subplots(1,1)
        plt.scatter(X,y)
        line, = plt.plot(plt_range, plt_range*m.val+b.val, color = 'black', label = 'gradient descent linear regression')
        clf = LinearRegression().fit(X,y)
        plt.plot(plt_range, clf.coef_*plt_range+clf.intercept_, alpha = 0.3, label = 'sklearn linear regression')
        plt.xlabel("x value")
        plt.ylabel("y value")
        plt.legend()
        fig.canvas.draw()
        
    loss = Var(0)
    for ep in range(epochs):
        prev = loss
        m, b, loss = gradient_descent(X, y, loss_function, m, b, lr, forward)
        # check if absolute tolerance meets early stopping condition
        if (abs(loss.val - prev.val) < earlyStop):
            break
        if (canPlot):
            # update the plot to show 
            # change over each epoch
            line.set_ydata(plt_range*m.val+b.val)
            ax.set_title("Linear Regression Update epoch = {}\nMSE={:.2E}".format(ep, float(loss.val)))
            fig.canvas.draw()
            plt.pause(0.5)
    # return coefficient and intercept
    return m, b, loss
