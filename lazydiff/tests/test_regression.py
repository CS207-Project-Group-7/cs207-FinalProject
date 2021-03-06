import pytest
from pytest import approx
from lazydiff.vars import Var
from lazydiff import ops
from lazydiff import regression

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1)

dim = 1
X,y,true_coef = make_regression(n_samples = 100, n_features = dim, n_informative = dim, bias = 10, \
                                coef = True, noise = 0, random_state=1)
def test_MSE():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    loss = regression.MSE(X, y, m, b)
    assert loss.val == approx(np.sum((X.sum(axis=1)-y)**2)/X.shape[0])

def test_MSE_regularized():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    loss = regression.MSE_regularized(X, y, m, b)
    numeric_loss = np.sum((X.sum(axis=1)-y)**2)/(2*X.shape[0]) + np.linalg.norm(m.val,1)
    assert loss.val == approx(numeric_loss)

def test_lasso_loss():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    loss = regression.lasso_loss(X, y, m, b)
    numeric_loss = np.sum((X.sum(axis=1)-y)**2)/(2*X.shape[0]) + np.linalg.norm(m.val,1)
    assert loss.val == approx(numeric_loss)

def test_ridge_loss():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    loss = regression.ridge_loss(X,y,m,b)
    numeric_loss = np.sum((X.sum(axis=1)-y)**2) + np.linalg.norm(m.val)**2
    assert loss.val == approx(numeric_loss)

def test_elastic_loss():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    loss = regression.elastic_loss(X, y, m, b)
    numeric_loss = np.sum((X.sum(axis=1)-y)**2)/(2*X.shape[0]) + 0.5*np.linalg.norm(m.val,1) + 0.5*0.5*np.linalg.norm(m.val)**2
    assert loss.val == approx(numeric_loss)


def test_gradient_descent():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    new_m, new_b, new_loss = regression.gradient_descent(X, y, regression.MSE, m, b)
    numeric_mse = np.sum((X.sum(axis=1)-y)**2)/(X.shape[0])
    # manually doing gradient descent
    n = X.shape[0]
    m_grad = 2/n*np.sum(-X.reshape(-1)*(y-((m.val*X).sum(axis=1))))
    manual_m = m.val-0.1*m_grad
    b_grad = 2/n*np.sum(-1*(y-((m.val*X).sum(axis=1)+b.val)))
    manual_b = b.val-0.1*b_grad

    assert new_loss.val == approx(numeric_mse)
    assert new_m.val == approx(manual_m, abs = 1e-3)
    assert new_b.val == approx(manual_b, abs = 1e-3)

def test_lasso():
    m_lasso = Var(np.ones(X.shape[1]))
    b_lasso = Var(0)
    earlyStop = 0 #1e-8
    forward = True
    epochs = 300
    lr = 0.1
    m_lasso, b_lasso, loss = regression.iterative_regression(X, y, m_lasso, b_lasso, regression.lasso_loss, lr, epochs,
                                                        earlyStop, forward)
    clf_l1 = Lasso().fit(X,y)
    assert m_lasso.val == approx(clf_l1.coef_, abs=1e-3)
    assert b_lasso.val == approx(clf_l1.intercept_, abs=1e-3)

def test_ridge():
    m_ridge = Var(np.ones(X.shape[1]))
    b_ridge = Var(0)
    earlyStop = 0 #1e-8
    forward = False
    epochs = 300
    lr = 0.001
    m_ridge, b_ridge, loss = regression.iterative_regression(X, y, m_ridge, b_ridge, regression.ridge_loss, lr, epochs,
                                                        earlyStop, forward)
    clf_l2 = Ridge().fit(X,y)
    assert m_ridge.val == approx(clf_l2.coef_, abs=1e-3)
    assert b_ridge.val == approx(clf_l2.intercept_, abs=1e-3)

def test_elastic():
    m_el = Var(np.ones(X.shape[1]))
    b_el = Var(0)
    earlyStop = 0 #1e-8
    forward = False
    epochs = 300
    lr = 0.1
    m_el, b_el, loss = regression.iterative_regression(X, y, m_el, b_el, regression.elastic_loss, lr, epochs,
                                                        earlyStop, forward)
    clf_el = ElasticNet().fit(X,y)
    assert m_el.val == approx(clf_el.coef_, abs=1e-3)
    assert b_el.val == approx(clf_el.intercept_, abs=1e-3)

def test_polynomial():
    X = np.linspace(-10,10,20).reshape(-1,1)
    y = X.reshape(-1)**2
    X = PolynomialFeatures(2, include_bias = False).fit_transform(X)
    m = Var(np.random.rand(X.shape[1]), seed = 1.0)
    b = Var(0)
    earlyStop = 0
    forward = False
    m, b, loss = regression.iterative_regression(X, y, m, b, regression.MSE, 0.0001, 10000,
                                                        earlyStop, forward)
    predict = np.sum(m.val*X, axis=1)+b.val
    assert predict == approx(y, abs=1e-2)

def test_store():
    m = Var(np.ones(X.shape[1]))
    b = Var(0)
    earlyStop = 1e-8
    forward = True
    history = {}
    m, b, loss = regression.iterative_regression(X, y, m, b, regression.MSE, 0.1, 100,
                                                    earlyStop, forward, history)
    clf = LinearRegression().fit(X,y)
    assert m.val == approx(clf.coef_, abs=1e-3)
    assert b.val == approx(clf.intercept_, abs=1e-3)
    assert history['m'][-1] == m.val
    assert history['b'][-1] == b.val
    assert history['loss'][-1] == loss.val
