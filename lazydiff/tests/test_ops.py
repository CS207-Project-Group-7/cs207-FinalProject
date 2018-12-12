import pytest
from lazydiff import ops
from lazydiff.vars import Var
import numpy as np

def test_sin():
    var1 = Var(np.pi)
    var2 = ops.sin(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([-1])

def test_cos():
    var1 = Var(np.pi)
    var2 = ops.cos(var1)
    var2.backward()
    assert var2.val == pytest.approx(-1)
    assert np.array([var2.grad(var1)]) == pytest.approx([0])

def test_tan():
    var1 = Var(0.)
    var2 = ops.tan(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_asin():
    var1 = Var(0.)
    var2 = ops.arcsin(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_acos():
    var1 = Var(0.)
    var2 = ops.arccos(var1)
    var2.backward()
    assert var2.val == pytest.approx(1.570796,abs=1e-2)
    assert var2.grad(var1) == np.array([-1])

def test_atan():
    var1 = Var(0.)
    var2 = ops.arctan(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_sinh():
    var1 = Var(0.)
    var2 = ops.sinh(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_cosh():
    var1 = Var(0.)
    var2 = ops.cosh(var1)
    var2.backward()
    assert var2.val == pytest.approx(1)
    assert var2.grad(var1) == np.array([0])

def test_tanh():
    var1 = Var(0.)
    var2 = ops.tanh(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_asinh():
    var1 = Var(0.)
    var2 = ops.arcsinh(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_acosh():
    var1 = Var(2.)
    var2 = ops.arccosh(var1)
    var2.backward()
    assert var2.val == np.arccosh(2.)
    assert var2.grad(var1) == np.array([1 / np.sqrt(3)])

def test_atanh():
    var1 = Var(0.)
    var2 = ops.arctanh(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_exp():
    var1 = Var(0)
    var2 = ops.exp(var1)
    var2.backward()
    assert var2.val == pytest.approx(1)
    assert var2.grad(var1) == np.array([1])

def test_log():
    var1 = Var(1.)
    var2 = ops.log(var1)
    var2.backward()
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([1])

def test_logistic():
    var1 = Var(0)
    var2 = ops.logistic(var1)
    var2.backward()
    assert var2.val == pytest.approx(0.5)
    assert var2.grad(var1) == var2.val * (1 - var2.val)

def test_sqrt():
    var1 = Var(4)
    var2 = ops.sqrt(var1)
    var2.backward()
    assert var2.val == pytest.approx(2)
    assert var2.grad(var1) == .5 * 1 / var2.val

def test_neg():
    var1 = Var(1)
    var2 = ops.neg(var1)
    var2.backward()
    assert var2.val == pytest.approx(-1)
    assert var2.grad(var1) == -1

def test_add():
    var1 = Var(1)
    var2 = Var(1)
    var3 = ops.add(var1, var2)
    var3.backward()
    assert var3.val == pytest.approx(2)
    assert var3.grad(var1) == 1.
    assert var3.grad(var2) == 1.

def test_sub():
    var1 = Var(1)
    var2 = Var(1)
    var3 = ops.sub(var1, var2)
    var3.backward()
    assert var3.val == pytest.approx(0)
    assert var3.grad(var1) == 1.
    assert var3.grad(var2) == -1

def test_mul():
    var1 = Var(1)
    var2 = Var(1)
    var3 = ops.mul(var1, var2)
    var3.backward()
    assert var3.val == pytest.approx(1)
    assert var3.grad(var1) == var2.val
    assert var3.grad(var2) == var1.val

def test_div():
    var1 = Var(1)
    var2 = Var(1)
    var3 = ops.div(var1, var2)
    var3.backward()
    assert var3.val == pytest.approx(1)
    assert var3.grad(var1) == 1.
    assert var3.grad(var2) == -1.

def test_pow():
    var1 = Var(1)
    var2 = Var(1)
    var3 = ops.pow(var1, var2)
    var3.backward()
    assert var3.val == pytest.approx(1)
    assert var3.grad(var1) == 1
    assert var3.grad(var2) == 0

def test_abs():
    var1 = Var(-1)
    var2 = ops.abs(var1)
    var2.backward()
    assert var2.val == 1.
    assert var2.grad(var1) == -1

def test_composite_logexp():
    x = Var(5)
    y = ops.log(ops.exp(x))
    y.backward()
    assert np.all(x.val == pytest.approx(y.val))
    assert np.all(y.grad(x) == 1)

def test_composite_trig():
    x = Var(5)
    x2 = ops.sin(x) / ops.cos(x)
    x3 = ops.tan(x)
    x.forward()
    assert np.all(x2.val == pytest.approx(x3.val))
    assert np.all(x2.grad(x) == pytest.approx(x3.grad(x)))
