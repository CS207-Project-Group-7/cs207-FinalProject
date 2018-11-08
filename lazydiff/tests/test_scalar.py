import pytest
import numpy as np
from lazydiff.vars import Scalar

def test_init_scalar():
    var = Scalar(1)
    assert var.val == 1

def test_grad_no_args():
    var = Scalar(1)
    with pytest.raises(ValueError):
        var.grad()

def test_vector_grad_invalid_arg_raises_error():
    var = Scalar(2)
    with pytest.raises(TypeError):
        var.grad('blah')

def test_neg():
    var1 = Scalar(2)
    var2 = -var1
    assert var2.val == -2
    assert var2.grad(var1) == np.array([-1])

def test_abs():
    var1 = Scalar(-2)
    var2 = abs(var1)
    assert var2.val == 2
    assert var2.grad(var1) == np.array([-1])

def test_add_scalars():
    var1 = Scalar(1)
    var2 = Scalar(2)
    var3 = var1 + var2
    assert var3.val == 3
    assert np.all(var3.grad(var1) == np.array([1]))
    assert np.all(var3.grad(var2) == np.array([1]))

def test_add_scalar_number():
    var1 = Scalar(3)
    var2 = var1 + 5
    assert var2.val == 8
    assert np.all(var2.grad(var1) == np.array([1]))

def test_radd_scalar():
    var1 = Scalar(3)
    var2 = 6 + var1
    assert var2.val == 9
    assert np.all(var2.grad(var1) == np.array([1]))

def test_add_scalar_non_number():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var + 'string'

def test_sub_scalars():
    var1 = Scalar(1)
    var2 = Scalar(2)
    var3 = var1 - var2
    assert var3.val == -1
    assert np.all(var3.grad(var1) == np.array([1]))
    assert np.all(var3.grad(var2) == np.array([-1]))

def test_sub_scalar_number():
    var1 = Scalar(3)
    var2 = var1 - 5
    assert var2.val == -2
    assert np.all(var2.grad(var1) == np.array([1]))

def test_rsub_scalar():
    var1 = Scalar(3)
    var2 = 6 - var1
    assert var2.val == 3
    assert np.all(var2.grad(var1) == np.array([-1]))

def test_sub_scalar_non_number():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var - 'string'

def test_mul_scalars():
    var1 = Scalar(8)
    var2 = Scalar(2)
    var3 = var1 * var2
    assert var3.val == 16
    assert np.all(var3.grad(var1) == np.array([2]))
    assert np.all(var3.grad(var2) == np.array([8]))

def test_mul_scalar_number():
    var1 = Scalar(3)
    var2 = var1 * 5
    assert var2.val == 15
    assert np.all(var2.grad(var1) == np.array([5]))

def test_rmul_scalars():
    var1 = Scalar(3)
    var2 = 6 * var1
    assert var2.val == 18
    assert np.all(var2.grad(var1) == np.array([6]))

def test_mul_scalar_non_number():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var * 'string'

def test_div_scalars():
    var1 = Scalar(8)
    var2 = Scalar(2)
    var3 = var1 / var2
    assert var3.val == 4
    assert np.all(var3.grad(var1) == np.array([0.5]))
    assert np.all(var3.grad(var2) == np.array([-2]))

def test_div_scalar_number():
    var1 = Scalar(10)
    var2 = var1 / 5
    assert var2.val == 2.
    assert np.all(var2.grad(var1) == np.array([0.2]))

def test_div_scalar_fails_with_divide_by_zero():
    vec = Scalar(1)
    with pytest.raises(ZeroDivisionError):
        vec / Scalar(0.)

def test_div_scalar_number_fails_with_divide_by_zero():
    vec = Scalar(1)
    with pytest.raises(ZeroDivisionError):
        vec / 0.

def test_rdiv_scalars():
    var1 = Scalar(3)
    var2 = 6 / var1
    assert var2.val == 2
    assert np.all(var2.grad(var1) == np.array([-2 / 3]))

def test_rdiv_scalar_fails_with_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / Scalar(0.)

def test_div_scalar_non_number():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var / 'string'

def test_pow_scalars():
    var1 = Scalar(np.e)
    var2 = Scalar(2)
    var3 = var1 ** var2
    assert var3.val == np.e ** 2
    assert np.all(var3.grad(var1) == np.array([2 * np.e]))
    assert np.all(var3.grad(var2) == np.array([np.e ** 2]))

def test_pow_scalar_number():
    var1 = Scalar(3)
    var2 = var1 ** 5
    assert var2.val == 243
    assert np.all(var2.grad(var1) == np.array([405]))

def test_rpow_scalars():
    var1 = Scalar(3)
    var2 = np.e ** var1
    assert var2.val == np.e ** 3
    assert np.all(var2.grad(var1) == np.array([np.e ** 3]))

def test_pow_scalar_non_number():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var ** 'string'

def test_iadd_banned():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var += 3

def test_isub_banned():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var -= 3

def test_imul_banned():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var *= 3

def test_idiv_banned():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var /= 3

def test_ipow_banned():
    var = Scalar(3)
    with pytest.raises(TypeError):
        var **= 3

def test_composition():
    var1 = Scalar(1)
    var2 = Scalar(2)
    var3 = Scalar(4)
    var4 = Scalar(3)
    var5 = var1 + var2
    var6 = var5 * var3
    var7 = var6 ** var4
    assert var7.val == 1728
    assert np.all(var7.grad(var1, var3, var5, var6) == np.array([1728, 1296, 1728, 432]))