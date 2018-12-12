from lazydiff.vars import Var
from lazydiff import ops
import numpy as np
import time

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

def iterative_regression(X, y, m, b, loss_function, lr = 0.1,\
        epochs = 100, earlyStop = 0, forward = True, history = None):
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
    history to store old values of m, b, loss 
    if provided a dictionary
    """
    canStore = isinstance(history, dict)
    
    if (canStore):
        history['m'] = []
        history['b'] = []
        history['loss'] = []

    loss = Var(0)
    for ep in range(epochs):
        prev = loss
        m, b, loss = gradient_descent(X, y, loss_function, m, b, lr, forward)
        if (canStore):
            # store the m, b
            # change over each epoch
            history['m'].append(m.val)
            history['b'].append(b.val)
            history['loss'].append(loss.val)

        # check if absolute tolerance meets early stopping condition
        if (abs(loss.val - prev.val) < earlyStop):
            break
    # return coefficient and intercept
    return m, b, loss
