import pytest
from lazydiff import ops
from lazydiff.vars import Var
import numpy as np

def test_sin():
    var1 = Var([np.pi, np.pi])
    var2 = ops.sin(var1)
    var2.backward()
    assert var2.val == pytest.approx([0, 0])
    assert np.all(var2.grad(var1) == np.array([-1, -1]))

def test_cos():
    var1 = Var([np.pi, np.pi])
    var2 = ops.cos(var1)
    var2.backward()
    assert var2.val == pytest.approx([-1, -1])
    assert np.array(var2.grad(var1)) == pytest.approx([0, 0])

def test_tan():
    var1 = Var([0, 0])
    var2 = ops.tan(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_asin():
    var1 = Var([0, 0])
    var2 = ops.arcsin(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_acos():
    var1 = Var([0, 0])
    var2 = ops.arccos(var1)
    var2.backward()
    assert np.all(var2.val == np.arccos([0, 0]))
    assert np.all(var2.grad(var1) == [-1, -1])

def test_atan():
    var1 = Var([0, 0])
    var2 = ops.arctan(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_sinh():
    var1 = Var([0, 0])
    var2 = ops.sinh(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_cosh():
    var1 = Var([0, 0])
    var2 = ops.cosh(var1)
    var2.backward()
    assert np.all(var2.val == [1, 1])
    assert np.all(var2.grad(var1) == [0, 0])

def test_tanh():
    var1 = Var([0, 0])
    var2 = ops.tanh(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_asinh():
    var1 = Var([0, 0])
    var2 = ops.arcsinh(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_acosh():
    var1 = Var([2, 2])
    var2 = ops.arccosh(var1)
    var2.backward()
    assert np.all(var2.val == np.arccosh([2, 2]))
    assert np.all(var2.grad(var1) == np.array([1, 1]) / np.sqrt(3))

def test_atanh():
    var1 = Var([0, 0])
    var2 = ops.arctanh(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_exp():
    var1 = Var([0, 0])
    var2 = ops.exp(var1)
    var2.backward()
    assert np.all(var2.val == [1, 1])
    assert np.all(var2.grad(var1) == [1, 1])

def test_log():
    var1 = Var([1., 1.])
    var2 = ops.log(var1)
    var2.backward()
    assert np.all(var2.val == [0, 0])
    assert np.all(var2.grad(var1) == [1, 1])

def test_logistic():
    var1 = Var([0, 0])
    var2 = ops.logistic(var1)
    var2.backward()
    assert np.all(var2.val == [.5, .5])
    assert np.all(var2.grad(var1) == (var2.val * (1 - var2.val)))

def test_sqrt():
    var1 = Var([4, 4])
    var2 = ops.sqrt(var1)
    var2.backward()
    assert np.all(var2.val == [2, 2])
    assert np.all(var2.grad(var1) == [.5 * 1 / var2.val, .5 * 1 / var2.val])

def test_neg():
    var1 = Var([1, 1])
    var2 = ops.neg(var1)
    var2.backward()
    assert np.all(var2.val == [-1, -1])
    assert np.all(var2.grad(var1) == [-1, -1])

def test_add():
    var1 = Var([1, 1])
    var2 = Var([1, 1])
    var3 = ops.add(var1, var2)
    var3.backward()
    assert np.all(var3.val == [2, 2])
    assert np.all(var3.grad(var1) == [1, 1])
    assert np.all(var3.grad(var2) == [1, 1])

def test_sub():
    var1 = Var([1, 1])
    var2 = Var([1, 1])
    var3 = ops.sub(var1, var2)
    var3.backward()
    assert np.all(var3.val == [0, 0])
    assert np.all(var3.grad(var1) == [1, 1])
    assert np.all(var3.grad(var2) == [-1, -1])

def test_mul():
    var1 = Var([1, 1])
    var2 = Var([1, 1])
    var3 = ops.mul(var1, var2)
    var3.backward()
    assert np.all(var3.val == [1, 1])
    assert np.all(var3.grad(var1) == var2.val)
    assert np.all(var3.grad(var2) == var1.val)

def test_div():
    var1 = Var([1, 1])
    var2 = Var([1, 1])
    var3 = ops.div(var1, var2)
    var3.backward()
    assert np.all(var3.val == [1, 1])
    assert np.all(var3.grad(var1) == [1, 1])
    assert np.all(var3.grad(var2) == [-1, -1])

def test_pow():
    var1 = Var([1, 1])
    var2 = Var([1, 1])
    var3 = ops.pow(var1, var2)
    var3.backward()
    assert np.all(var3.val == [1, 1])
    assert np.all(var3.grad(var1) == [1, 1])
    assert np.all(var3.grad(var2) == [0, 0])

def test_abs():
    var1 = Var([-1, -1])
    var2 = ops.abs(var1)
    var2.backward()
    assert np.all(var2.val == [1, 1])
    assert np.all(var2.grad(var1) == [-1, -1])

def test_sum():
    var1 = Var([2, 2, 2, 2, 2])
    var2 = ops.sum(var1)
    var2.backward()
    assert var2.val == 10.
    assert np.all(var2.grad(var1) == [1, 1, 1, 1, 1])

def test_norm():
    var1 = Var([1, 2, 3])
    var2 = ops.norm(var1, p=2)
    var2.backward()
    assert var2.val == np.linalg.norm(var1.val)
    assert np.all(var2.grad(var1) == [1/np.sqrt(14), np.sqrt(2/7), 3/np.sqrt(14)])

def test_composite_logexp():
    x = Var([5, 10, 15, 20])
    y = ops.log(ops.exp(x))
    y.backward()
    assert np.all(x == y)
    assert np.all(y.grad(x) == 1)

def test_composite_trig():
    x = Var([5, 10, 15, 20])
    x2 = ops.sin(x) / ops.cos(x)
    x3 = ops.tan(x)
    x.forward()
    assert np.all(x2.val == pytest.approx(x3.val))
    assert np.all(x2.grad(x) == x3.grad(x))
