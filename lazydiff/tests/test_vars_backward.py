import pytest
import numpy as np
from lazydiff.vars import Var

def test_init_var():
    var = Var(1)
    assert var.val == 1

def test_init_var_vector():
    var = Var([1, 2, 3])
    assert np.all(var.val == [1, 2, 3])

def test_vars_equal():
    x1 = Var(1)
    x2 = Var(1)
    x3 = Var(2)
    assert x1 == x2
    assert not x1 == x3

def test_vars_nequal():
    x1 = Var(1)
    x2 = Var(2)
    x3 = Var(1)
    assert x1 != x2
    assert not x1 != x3

def test_vars_lessthan():
    x1 = Var(1)
    x2 = Var(2)
    assert x1 < x2
    assert not x2 < x1

def test_vars_greaterthan():
    x1 = Var(2)
    x2 = Var(1)
    assert x1 > x2
    assert not x2 > x1

def test_vars_le():
    x1 = Var(1)
    x2 = Var(2)
    x3 = Var(1)
    assert x1 <= x2
    assert x1 == x3
    assert not x2 <= x1

def test_vars_ge():
    x1 = Var(2)
    x2 = Var(1)
    assert x1 >= x2
    assert not x2 >= x1

def test_comp_non_var():
    x1 = Var(2)
    assert x1 != 2
    assert x1 != [2]
    assert x1 != np.array(2)

def test_invalid_arg_raises_error():
    var = Var(2)
    with pytest.raises(TypeError):
        var.grad('invalid value')

def test_neg():
    x = Var(2)
    y = -x
    assert y.val == -2
    y.backward()
    assert y.grad(x) == np.array([-1])

def test_abs():
    x = Var(-2)
    y = abs(x)
    y.backward()
    assert y.val == 2
    assert y.grad(x) == np.array([-1])

def test_add_vars():
    x1 = Var(1)
    x2 = Var(2)
    y = x1 + x2
    y.backward()
    assert y.val == 3
    assert np.all(y.grad(x1) == np.array([1]))
    assert np.all(y.grad(x2) == np.array([1]))

def test_add_var_number():
    x = Var(3)
    y = x + 5
    y.backward()
    assert y.val == 8
    assert np.all(y.grad(x) == np.array([1]))

def test_radd_var():
    x = Var(3)
    y = 6 + x
    y.backward()
    assert y.val == 9
    assert np.all(y.grad(x) == np.array([1]))

def test_add_var_non_number():
    with pytest.raises(TypeError):
        Var(3) + 'string'

def test_sub_vars():
    x1 = Var(1)
    x2 = Var(2)
    y = x1 - x2
    y.backward()
    assert y.val == -1
    assert np.all(y.grad(x1) == np.array([1]))
    assert np.all(y.grad(x2) == np.array([-1]))

def test_sub_var_number():
    x = Var(3)
    y = x - 5
    y.backward()
    assert y.val == -2
    assert np.all(y.grad(x) == np.array([1]))

def test_rsub_var():
    x = Var(3)
    y = 6 - x
    y.backward()
    assert y.val == 3
    assert np.all(y.grad(x) == np.array([-1]))

def test_sub_var_non_number():
    with pytest.raises(TypeError):
        Var(3) - 'string'

def test_mul_vars():
    x1 = Var(8)
    x2 = Var(2)
    y = x1 * x2
    y.backward()
    assert y.val == 16
    assert np.all(y.grad(x1) == np.array([2]))
    assert np.all(y.grad(x2) == np.array([8]))

def test_mul_var_number():
    x = Var(3)
    y = x * 5
    y.backward()
    assert y.val == 15
    assert np.all(y.grad(x) == np.array([5]))

def test_rmul_vars():
    x = Var(3)
    y = 6 * x
    y.backward()
    assert y.val == 18
    assert np.all(y.grad(x) == np.array([6]))

def test_mul_var_non_number():
    with pytest.raises(TypeError):
        Var(1) * 'string'

def test_div_vars():
    x = Var(8)
    x2 = Var(2)
    y = x / x2
    y.backward()
    assert y.val == 4
    assert np.all(y.grad(x) == np.array([0.5]))
    assert np.all(y.grad(x2) == np.array([-2]))

def test_div_var_number():
    x = Var(10)
    y = x / 5
    y.backward()
    assert y.val == 2.
    assert np.all(y.grad(x) == np.array([0.2]))

def test_div_var_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        Var(1) / Var(0.)

def test_div_var_number_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        Var(1) / 0.

def test_rdiv_vars():
    x = Var(3)
    y = 6 / x
    y.backward()
    assert y.val == 2
    assert np.all(y.grad(x) == np.array([-2 / 3]))

def test_rdiv_var_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        1 / Var(0.)

def test_div_var_non_number():
    with pytest.raises(TypeError):
        Var(1) / 'string'

def test_pow_vars():
    x = Var(np.e)
    x2 = Var(2)
    y = x ** x2
    y.backward()
    assert y.val == np.e ** 2
    assert np.all(y.grad(x) == np.array([2 * np.e]))
    assert np.all(y.grad(x2) == np.array([np.e ** 2]))

def test_pow_var_number():
    x = Var(3)
    y = x ** 5
    y.backward()
    assert y.val == 243
    assert np.all(y.grad(x) == np.array([405]))

def test_rpow_vars():
    x = Var(3)
    y = np.e ** x
    y.backward()
    assert y.val == np.e ** 3
    assert np.all(y.grad(x) == np.array([np.e ** 3]))

def test_pow_var_non_number():
    x = Var(1)
    with pytest.raises(TypeError):
        x **= 'string'

def test_iadd_banned():
    x = Var(1)
    with pytest.raises(TypeError):
        x += 3

def test_isub_banned():
    x = Var(1)
    with pytest.raises(TypeError):
        x -= 3

def test_imul_banned():
    x = Var(1)
    with pytest.raises(TypeError):
        x *= 3

def test_idiv_banned():
    x = Var(1)
    with pytest.raises(TypeError):
        x /= 3

def test_ipow_banned():
    x = Var(1)
    with pytest.raises(TypeError):
        x **= 3

def test_repr():
    var = Var(1)
    expected = 'Var(1.0, seed=1.0)'
    assert repr(var) == expected

def test_vars_not_linked_fails():
    x = Var(1)
    y = Var(1)
    with pytest.raises(ValueError):
        y.backward()
        y.grad(x)
