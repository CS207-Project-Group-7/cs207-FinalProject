import pytest
import numpy as np
from lazydiff.vars import Var

def test_init_var():
    var = Var([1, 2, 3])
    assert np.all(var.val == [1, 2, 3])

def test_invalid_arg_raises_error():
    var = Var([1, 2, 3])
    with pytest.raises(TypeError):
        var.grad('invalid value')

def test_neg():
    x = Var([1, 2, 3])
    y = -x
    y.backward()
    assert np.all(y == Var([-1, -2, -3]))
    assert np.all(y.grad(x) == np.array([-1, -1, -1]))

def test_abs():
    x = Var([1, 2, -3])
    y = abs(x)
    y.backward()
    assert np.all(y == Var([1, 2, 3]))
    assert np.all(y.grad(x) == np.array([1, 1, -1]))

def test_add_vars():
    x1 = Var([1, 2, 3])
    x2 = Var([1, 2, 3])
    y = x1 + x2
    y.backward()
    assert np.all(y == Var([2, 4, 6]))
    assert np.all(y.grad(x1) == np.array([1, 1, 1]))
    assert np.all(y.grad(x2) == np.array([1, 1, 1]))

def test_add_var_number():
    x = Var([1, 2, 3])
    y = x + 5
    y.backward()
    assert np.all(y == Var([6, 7, 8]))
    assert np.all(y.grad(x) == np.array([1, 1, 1]))

def test_radd_var():
    x = Var([1, 2, 3])
    y = 6 + x
    y.backward()
    assert np.all(y == Var([7, 8, 9]))
    assert np.all(y.grad(x) == np.array([1, 1, 1]))

def test_add_var_non_number():
    with pytest.raises(TypeError):
        Var([1, 2, 3]) + 'string'

def test_sub_vars():
    x1 = Var([1, 1, 1])
    x2 = Var([1, 2, 3])
    y = x1 - x2
    y.backward()
    assert np.all(y == Var([0, -1, -2]))
    assert np.all(y.grad(x1) == np.array([1, 1, 1]))
    assert np.all(y.grad(x2) == np.array([-1, -1, -1]))

def test_sub_var_number():
    x = Var([1, 2, 3])
    y = x - 5
    y.backward()
    assert np.all(y == Var([-4, -3, -2]))
    assert np.all(y.grad(x) == np.array([1, 1, 1]))

def test_rsub_var():
    x = Var([1, 2, 3])
    y = 6 - x
    y.backward()
    assert np.all(y == Var([5, 4, 3]))
    assert np.all(y.grad(x) == np.array([-1, -1, -1]))

def test_sub_var_non_number():
    with pytest.raises(TypeError):
        Var([1, 2, 3]) - 'string'

def test_mul_vars():
    x1 = Var(8)
    x2 = Var([2, 2, 2])
    y = x1 * x2
    y.backward()
    assert np.all(y == Var([16, 16, 16]))
    assert np.all(y.grad(x1) == np.array([2, 2, 2]))
    assert np.all(y.grad(x2) == np.array([8, 8, 8]))

def test_mul_var_number():
    x = Var([3, 2, 1])
    y = x * 5
    y.backward()
    assert np.all(y == Var([15, 10, 5]))
    assert np.all(y.grad(x) == np.array([5, 5, 5]))

def test_rmul_vars():
    x = Var([3, 3, 3])
    y = 6 * x
    y.backward()
    assert np.all(y == Var([18, 18, 18]))
    assert np.all(y.grad(x) == np.array([6, 6, 6]))

def test_mul_var_non_number():
    with pytest.raises(TypeError):
        Var([1, 2, 3]) * 'string'

def test_div_vars():
    x = Var([8, 1])
    x2 = Var([2, 1])
    y = x / x2
    y.backward()
    assert np.all(y == Var([4, 1]))
    assert np.all(y.grad(x) == np.array([0.5, 1]))
    assert np.all(y.grad(x2) == np.array([-2., -1]))

def test_div_var_number():
    x = Var([10, 10])
    y = x / 5
    y.backward()
    assert np.all(y == Var([2, 2]))
    assert np.all(y.grad(x) == np.array([0.2, .2]))

def test_div_var_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        Var([1, 2, 3]) / Var(0.)

def test_div_var_number_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        Var([1, 2, 3]) / 0.

def test_rdiv_vars():
    x = Var([3, 3, 3])
    y = 6 / x
    y.backward()
    assert np.all(y == Var([2, 2, 2]))
    assert np.all(y.grad(x) == np.array([-2/3, -2/3, -2/3]))

def test_rdiv_var_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        1 / Var([0, 2, 3])

def test_div_var_non_number():
    with pytest.raises(TypeError):
        Var([1, 2, 3]) / 'string'

def test_pow_vars():
    x = Var(np.e)
    x2 = Var([1, 2, 3])
    y = x ** x2
    y.backward()
    assert np.all(y == Var([np.e**1, np.e**2, np.e**3]))
    assert np.all(y.grad(x) == np.array([1, 2*np.e, 3*np.e**2]))
    assert np.all(y.grad(x2) == np.array([np.e, np.e**2, np.e**3]))

def test_pow_var_number():
    x = Var([1, 2, 3])
    y = x ** 5
    y.backward()
    assert np.all(y == Var([1, 32, 243]))
    assert np.all(y.grad(x) == np.array([5, 80, 405]))

def test_rpow_vars():
    x = Var([1, 2, 3])
    y = np.e ** x
    y.backward()
    assert np.all(y == Var([np.e, np.e**2, np.e**3]))
    assert np.all(y.grad(x) == np.array([np.e, np.e**2, np.e**3]))

def test_pow_var_non_number():
    x = Var([1, 2, 3])
    with pytest.raises(TypeError):
        x **= 'string'

def test_iadd_banned():
    x = Var([1, 2, 3])
    with pytest.raises(TypeError):
        x += 3

def test_isub_banned():
    x = Var([1, 2, 3])
    with pytest.raises(TypeError):
        x -= 3

def test_imul_banned():
    x = Var([1, 2, 3])
    with pytest.raises(TypeError):
        x *= 3

def test_idiv_banned():
    x = Var([1, 2, 3])
    with pytest.raises(TypeError):
        x /= 3

def test_ipow_banned():
    x = Var([1, 2, 3])
    with pytest.raises(TypeError):
        x **= 3

def test_composite1():
    x1 = Var(-3)
    x2 = Var([5, 10])
    x3 = Var([10, 100])
    y = abs(x1) / x2 * x3
    y.backward()
    assert np.all(y.grad(x1) == [-2, -10])

def test_composite2():
    x1 = Var(-3)
    x2 = Var([5, 10])
    x3 = Var([1, 1])
    y = x2**x1 / x3
    y.backward()
    assert np.all(y.grad(x3) == [-.008, -.001])
