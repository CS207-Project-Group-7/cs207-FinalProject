import pytest
from lazydiff.vars import Vector, Scalar
import numpy as np

def test_init_vector_with_scalar():
    vec = Vector(Scalar(7.))
    assert vec.val == (7.,)

def test_init_vector_with_scalars():
    vec = Vector(Scalar(7.), Scalar(9.))
    assert vec.val == (7., 9.)

def test_init_vector_with_ndarray():
    vec = Vector(np.array([Scalar(7.), Scalar(9.)]))
    assert vec.val == (7., 9.)

def test_init_vector_fails_with_non_scalar():
    with pytest.raises(TypeError):
        Vector(7.)
    with pytest.raises(TypeError):
        Vector([7.])
    with pytest.raises(TypeError):
        Vector(np.array([7.]))

def test_vector_grad_empty_raises_error():
    vec = Vector([Scalar(2.), Scalar(3.)])
    with pytest.raises(ValueError):
        vec.grad()

def test_vector_grad_scalar():
    val1 = Scalar(2.)
    vec = Vector([val1, Scalar(3.)])
    assert np.all(vec.grad(val1) == np.array([[1.], [0,]]))

def test_vector_grad_vector():
    var1 = Scalar(2.)
    var2 = Scalar(3.)
    vec = Vector([var1, var2])
    assert np.all(vec.grad(var1, var2) == np.array([[1., 0], [0, 1.]]))

def test_get_vector_item():
    vec = Vector([Scalar(2.), Scalar(3.)])
    assert vec[0].val == 2
    assert vec[1].val == 3

def test_get_vector_item_out_of_range_fails():
    vec = Vector([Scalar(2.), Scalar(3.)])
    with pytest.raises(IndexError):
        vec[2]
    with pytest.raises(IndexError):
        vec[-1]
    
def test_vector_negation():
    vec = Vector([Scalar(2.), Scalar(3.)])
    assert (-vec).val == (-2., -3.)

def test_vector_add_number():
    vec = Vector([Scalar(1.), Scalar(2.)])
    added = vec + 2.
    assert added.val == (3., 4.)

def test_vector_add_scalar():
    vec = Vector([Scalar(1.), Scalar(2.)])
    added = vec + Scalar(2.)
    assert added.val == (3., 4.)

def test_vector_add_vector():
    vec = Vector([Scalar(1.), Scalar(2.)])
    vec2 = Vector([Scalar(3.), Scalar(4.)])
    added = vec + vec2
    assert added.val == (4., 6.)

def test_vector_add_vector_wrong_shape_fails():
    vec = Vector([Scalar(1.), Scalar(2.)])
    vec2 = Vector([Scalar(3.), Scalar(4.), Scalar(5.)])
    with pytest.raises(ValueError):
        vec + vec2
    
def test_vector_add_vector_non_numeric():
    vec = Vector([Scalar(1.), Scalar(2.)])
    with pytest.raises(TypeError):
        vec + 'hello there'
    
def test_vector_radd():
    vec = Vector([Scalar(1.), Scalar(2.)])
    added = 2. + vec
    assert added.val == (3., 4.)

def test_vector_subtract():
    vec = Vector([Scalar(1.), Scalar(2.)])
    added = vec - 2.
    print(added.val)
    assert added.val == (-1., 0.)

def test_vector_rsub():
    vec = Vector([Scalar(1.), Scalar(2.)])
    added = 2. - vec
    print(added.val)
    assert added.val == (1., 0.)

def test_vector_mul_number():
    vec = Vector([Scalar(1.), Scalar(2.)])
    prod = vec * 2.
    assert prod.val == (2., 4.)

def test_vector_mul_scalar():
    vec = Vector([Scalar(1.), Scalar(2.)])
    prod = vec * Scalar(2.)
    assert np.all(prod.val == np.array([2., 4.]))
    assert np.all(prod.grad(vec) == np.array([[2., 0], [0, 2]]))

def test_vector_mul_vector():
    vec = Vector([Scalar(1.), Scalar(2.)])
    vec2 = Vector([Scalar(2.), Scalar(3.)])
    prod = vec * vec2
    assert prod.val == (2., 6.)

def test_vector_mul_vector_wrong_shape_fails():
    vec = Vector([Scalar(1.), Scalar(2.)])
    vec2 = Vector([Scalar(3.), Scalar(4.), Scalar(5.)])
    with pytest.raises(ValueError):
        vec * vec2
    
def test_vector_mul_vector_non_numeric():
    vec = Vector([Scalar(1.), Scalar(2.)])
    with pytest.raises(TypeError):
        vec * 'hello there'
    
def test_vector_rmul():
    vec = Vector([Scalar(1.), Scalar(2.)])
    prod = 2. * vec
    assert prod.val == (2., 4.)

def test_vector_truediv():
    vec = Vector([Scalar(1.), Scalar(2.)])
    divided = vec / Scalar(2.)
    assert divided.val == (.5, 1.)

def test_vector_truediv_fails_with_divide_by_zero():
    vec = Vector([Scalar(1.), Scalar(2.)])
    with pytest.raises(ValueError):
        vec / Scalar(0.)

def test_vector_rtruediv():
    vec = Vector([Scalar(1.), Scalar(2.)])
    divided = vec.__rtruediv__(Scalar(2.))
    assert divided.val == (2., 1.)

def test_vector_rtruediv_fails_with_divide_by_zero():
    vec = Vector([Scalar(0.), Scalar(0.)])
    with pytest.raises(ValueError):
        1. / vec

def test_vector_pow():
    vec = Vector([Scalar(1.), Scalar(2.)])
    powed = vec**2
    assert powed.val == (1., 4.)
