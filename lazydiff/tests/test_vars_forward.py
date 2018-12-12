import pytest
import numpy as np
from lazydiff.vars import Var

def test_init_var_forward():
    var = Var(1)
    assert var.val == 1

def test_init_var_vector_forward():
    var = Var([1, 2, 3])
    assert np.all(var.val == [1, 2, 3])

def test_neg_forward():
    x = Var(2)
    y = -x
    assert y.val == -2
    x.forward()
    assert y.grad(x) == np.array([-1])

def test_abs_forward():
    x = Var(-2)
    y = abs(x)
    x.forward()
    assert y.val == 2
    assert y.grad(x) == np.array([-1])

def test_add_vars_forward():
    x1 = Var(1)
    x2 = Var(2)
    y = x1 + x2
    x1.forward()
    x2.forward()
    assert y.val == 3
    assert np.all(y.grad(x1) == np.array([1]))
    assert np.all(y.grad(x2) == np.array([1]))

def test_add_var_number_forward():
    x = Var(3)
    y = x + 5
    x.forward()
    assert y.val == 8
    assert np.all(y.grad(x) == np.array([1]))

def test_radd_var_forward():
    x = Var(3)
    y = 6 + x
    x.forward()
    assert y.val == 9
    assert np.all(y.grad(x) == np.array([1]))

def test_sub_vars_forward():
    x1 = Var(1)
    x2 = Var(2)
    y = x1 - x2
    x1.forward()
    x2.forward()
    assert y.val == -1
    assert np.all(y.grad(x1) == np.array([1]))
    assert np.all(y.grad(x2) == np.array([-1]))

def test_sub_var_number_forward():
    x = Var(3)
    y = x - 5
    x.forward()
    assert y.val == -2
    assert np.all(y.grad(x) == np.array([1]))

def test_rsub_var_forward():
    x = Var(3)
    y = 6 - x
    x.forward()
    assert y.val == 3
    assert np.all(y.grad(x) == np.array([-1]))

def test_mul_vars_forward():
    x1 = Var(8)
    x2 = Var(2)
    y = x1 * x2
    x1.forward()
    x2.forward()
    assert y.val == 16
    assert np.all(y.grad(x1) == np.array([2]))
    assert np.all(y.grad(x2) == np.array([8]))

def test_mul_var_number_forward():
    x = Var(3)
    y = x * 5
    x.forward()
    assert y.val == 15
    assert np.all(y.grad(x) == np.array([5]))

def test_rmul_vars_forward():
    x = Var(3)
    y = 6 * x
    x.forward()
    assert y.val == 18
    assert np.all(y.grad(x) == np.array([6]))

def test_div_vars_forward():
    x = Var(8)
    x2 = Var(2)
    y = x / x2
    x.forward()
    x2.forward()
    assert y.val == 4
    assert np.all(y.grad(x) == np.array([0.5]))
    assert np.all(y.grad(x2) == np.array([-2]))

def test_div_var_number_forward():
    x = Var(10)
    y = x / 5
    x.forward()
    assert y.val == 2.
    assert np.all(y.grad(x) == np.array([0.2]))

def test_rdiv_vars_forward():
    x = Var(3)
    y = 6 / x
    x.forward()
    assert y.val == 2
    assert np.all(y.grad(x) == np.array([-2 / 3]))

def test_pow_vars_forward():
    x = Var(np.e)
    x2 = Var(2)
    y = x ** x2
    x.forward()
    x2.forward()
    assert y.val == np.e ** 2
    assert np.all(y.grad(x) == np.array([2 * np.e]))
    assert np.all(y.grad(x2) == np.array([np.e ** 2]))

def test_pow_var_number_forward():
    x = Var(3)
    y = x ** 5
    x.forward()
    assert y.val == 243
    assert np.all(y.grad(x) == np.array([405]))

def test_rpow_vars_forward():
    x = Var(3)
    y = np.e ** x
    x.forward()
    assert y.val == np.e ** 3
    assert np.all(y.grad(x) == np.array([np.e ** 3]))

def test_composite1():
    x1 = Var(-3)
    x2 = Var(5)
    x3 = Var(10)
    y = abs(x1) / x2 * x3
    x1.forward()
    assert y.grad(x1) == -2.

def test_composite2():
    x1 = Var(-3)
    x2 = Var(5)
    x3 = Var(1)
    y = x2**x1 / x3
    x3.forward()
    assert y.grad(x3) == -.008
