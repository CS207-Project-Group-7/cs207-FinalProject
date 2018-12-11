import pytest
import numpy as np
from lazydiff.vars import Var

def test_init_scalar():
    var = Var(1)
    assert var.val == 1

def test_grad_no_args():
    var = Var(1)
    with pytest.raises(ValueError):
        var.grad()

def test_vector_grad_invalid_arg_raises_error():
    var = Var(2)
    with pytest.raises(TypeError):
        var.grad('blah')

def test_neg():
    var1 = Var(2)
    var2 = -var1
    assert var2.val == -2
    assert var2.grad(var1) == np.array([-1])

def test_abs():
    var1 = Var(-2)
    var2 = abs(var1)
    assert var2.val == 2
    assert var2.grad(var1) == np.array([-1])

def test_add_scalars():
    var1 = Var(1)
    var2 = Var(2)
    var3 = var1 + var2
    assert var3.val == 3
    assert np.all(var3.grad(var1) == np.array([1]))
    assert np.all(var3.grad(var2) == np.array([1]))

def test_add_scalar_number():
    var1 = Var(3)
    var2 = var1 + 5
    assert var2.val == 8
    assert np.all(var2.grad(var1) == np.array([1]))

def test_radd_scalar():
    var1 = Var(3)
    var2 = 6 + var1
    assert var2.val == 9
    assert np.all(var2.grad(var1) == np.array([1]))

def test_add_scalar_non_number():
    var = Var(3)
    with pytest.raises(TypeError):
        var + 'string'

def test_sub_scalars():
    var1 = Var(1)
    var2 = Var(2)
    var3 = var1 - var2
    assert var3.val == -1
    assert np.all(var3.grad(var1) == np.array([1]))
    assert np.all(var3.grad(var2) == np.array([-1]))

def test_sub_scalar_number():
    var1 = Var(3)
    var2 = var1 - 5
    assert var2.val == -2
    assert np.all(var2.grad(var1) == np.array([1]))

def test_rsub_scalar():
    var1 = Var(3)
    var2 = 6 - var1
    assert var2.val == 3
    assert np.all(var2.grad(var1) == np.array([-1]))

def test_sub_scalar_non_number():
    var = Var(3)
    with pytest.raises(TypeError):
        var - 'string'

def test_mul_scalars():
    var1 = Var(8)
    var2 = Var(2)
    var3 = var1 * var2
    assert var3.val == 16
    assert np.all(var3.grad(var1) == np.array([2]))
    assert np.all(var3.grad(var2) == np.array([8]))

def test_mul_scalar_number():
    var1 = Var(3)
    var2 = var1 * 5
    assert var2.val == 15
    assert np.all(var2.grad(var1) == np.array([5]))

def test_rmul_scalars():
    var1 = Var(3)
    var2 = 6 * var1
    assert var2.val == 18
    assert np.all(var2.grad(var1) == np.array([6]))

def test_mul_scalar_non_number():
    var = Var(3)
    with pytest.raises(TypeError):
        var * 'string'

def test_div_scalars():
    var1 = Var(8)
    var2 = Var(2)
    var3 = var1 / var2
    assert var3.val == 4
    assert np.all(var3.grad(var1) == np.array([0.5]))
    assert np.all(var3.grad(var2) == np.array([-2]))

def test_div_scalar_number():
    var1 = Var(10)
    var2 = var1 / 5
    assert var2.val == 2.
    assert np.all(var2.grad(var1) == np.array([0.2]))

def test_div_scalar_fails_with_divide_by_zero():
    vec = Var(1)
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        vec / Var(0.)

def test_div_scalar_number_fails_with_divide_by_zero():
    vec = Var(1)
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        vec / 0.

def test_rdiv_scalars():
    var1 = Var(3)
    var2 = 6 / var1
    assert var2.val == 2
    assert np.all(var2.grad(var1) == np.array([-2 / 3]))

def test_rdiv_scalar_fails_with_divide_by_zero():
    with pytest.raises((ZeroDivisionError, FloatingPointError)):
        1 / Var(0.)

def test_div_scalar_non_number():
    var = Var(3)
    with pytest.raises(TypeError):
        var / 'string'

def test_pow_scalars():
    var1 = Var(np.e)
    var2 = Var(2)
    var3 = var1 ** var2
    assert var3.val == np.e ** 2
    assert np.all(var3.grad(var1) == np.array([2 * np.e]))
    assert np.all(var3.grad(var2) == np.array([np.e ** 2]))

def test_pow_scalar_number():
    var1 = Var(3)
    var2 = var1 ** 5
    assert var2.val == 243
    assert np.all(var2.grad(var1) == np.array([405]))

def test_rpow_scalars():
    var1 = Var(3)
    var2 = np.e ** var1
    assert var2.val == np.e ** 3
    assert np.all(var2.grad(var1) == np.array([np.e ** 3]))

def test_pow_scalar_non_number():
    var = Var(3)
    with pytest.raises(TypeError):
        var ** 'string'

def test_iadd_banned():
    var = Var(3)
    with pytest.raises(TypeError):
        var += 3

def test_isub_banned():
    var = Var(3)
    with pytest.raises(TypeError):
        var -= 3

def test_imul_banned():
    var = Var(3)
    with pytest.raises(TypeError):
        var *= 3

def test_idiv_banned():
    var = Var(3)
    with pytest.raises(TypeError):
        var /= 3

def test_ipow_banned():
    var = Var(3)
    with pytest.raises(TypeError):
        var **= 3

def test_composition():
    var1 = Var(1)
    var2 = Var(2)
    var3 = Var(4)
    var4 = Var(3)
    var5 = var1 + var2
    var6 = var5 * var3
    var7 = var6 ** var4
    assert var7.val == 1728
    assert np.all(var7.grad(var1, var3, var5, var6) == 
           [np.array([1728]), np.array([1296]), np.array([1728]), np.array([432])])