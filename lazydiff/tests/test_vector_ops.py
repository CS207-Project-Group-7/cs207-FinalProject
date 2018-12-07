import pytest
import math
from lazydiff.vars import Scalar
import lazydiff.ops as ops
import numpy as np
import pytest
from pytest import approx

def test_type_error_ops():
    with pytest.raises(TypeError):
        ops.sin(3)

def test_vec_sin():
    s1,s2 = Scalar(0), Scalar(0)
    vec = ops.sin(Vector(s1,s2))
    assert vec.val == approx((0,0))
    jac = vec.grad(s1,s2)
    assert jac[0][0] == approx(1)
    assert jac[1][1] == approx(1)

def test_vec_cos():
    vec = Vector([Scalar(math.pi), Scalar(0)])
    vec2 = ops.cos(vec)
    assert vec2.val == approx((-1, 1))
    jac = vec2.grad(vec)
    assert jac[0][0] == approx(0)
    assert jac[1][1] == approx(0)

def test_vec_tan():
    vec = Vector([Scalar(math.pi), Scalar(0)])
    vec2 = ops.tan(vec)
    assert vec2.val == approx((0, 0))
    jac = vec2.grad(vec)
    assert jac[0][0] == approx(1)
    assert jac[1][1] == approx(1)

def test_vec_arcsin():
    vec = Vector(Scalar(0))
    vec2 = ops.arcsin(vec)
    assert vec2.val[0] == approx(0)
    assert vec2.grad(vec) == approx(1)

def test_vec_arccos():
    vec = Vector(Scalar(0))
    vec2 = ops.arccos(vec)
    assert vec2.val[0] == approx(math.pi/2)
    assert vec2.grad(vec) == approx(-1)

def test_vec_atan():
    vec = Vector(Scalar(0))
    vec2 = ops.arctan(vec)
    assert vec2.val[0] == approx(0)
    assert vec2.grad(vec) == approx(1)

def test_vec_sinh():
    vec = Vector(Scalar(0))
    vec2 = ops.sinh(vec)
    assert vec2.val[0] == approx(0)
    assert vec2.grad(vec) == approx(1)

def test_vec_cosh():
    vec = Vector(Scalar(0))
    vec2 = ops.cosh(vec)
    assert vec2.val[0] == approx(1)
    assert vec2.grad(vec) == approx(0)

def test_vec_tanh():
    vec = Vector(Scalar(0))
    vec2 = ops.tanh(vec)
    assert vec2.val[0] == approx(0)
    assert vec2.grad(vec) == approx(1)

def test_vec_arcsinh():
    vec = Vector(Scalar(0))
    vec2 = ops.arcsinh(vec)
    assert vec2.val[0] == approx(0)
    assert vec2.grad(vec) == approx(1)

def test_vec_arccosh():
    vec = Vector(Scalar(np.pi))
    vec2 = ops.arccosh(vec)
    # from wolframalpha
    assert vec2.val[0] == approx(1.81152627)
    assert vec2.grad(vec) == approx(0.3357746)

def test_vec_arctanh():
    vec = Vector(Scalar(0))
    vec2 = ops.arctanh(vec)
    assert vec2.val[0] == approx(0)
    assert vec2.grad(vec) == approx(1)

def test_vec_exp():
    vec = Vector(Scalar(2))
    vec2 = ops.exp(vec)
    assert vec2.val[0] == approx(np.exp(2))
    assert vec2.grad(vec) == approx(np.exp(2))

def test_vec_log():
    vec = Vector(Scalar(3))
    vec2 = ops.log(vec)
    assert vec2.val[0] == approx(np.log(3))
    assert vec2.grad(vec) == approx(1/3)

def test_vec_log_base2():
    vec = Vector(Scalar(1), Scalar(2))
    vec2 = ops.log(vec,2)
    assert vec2.val == approx((0,1))
    jac = vec2.grad(vec)
    assert jac[0][0]== approx(1/math.log(2))
    assert jac[1][1] == approx(1/(2*math.log(2)))

def test_vec_abs():
    vec = Vector([Scalar(-i) for i in range(1,4)])
    vec2 = ops.abs(vec)
    assert vec2.val == approx((1,2,3))
    assert np.all(vec2.grad(vec)[i][i] == approx(-1) for i in range(3))

def test_chained_sin2x():
    vec = Vector(Scalar(0), Scalar(math.pi/2))
    vec2 = vec*2
    vec3 = ops.sin(vec2)
    assert vec3.val == approx((0,0))
    jac_wrt_vec = vec3.grad(vec)
    assert jac_wrt_vec[0][0] == approx(2)
    assert jac_wrt_vec[1][1] == approx(-2)
    jac_wrt_vec2 = vec3.grad(vec2)
    assert jac_wrt_vec2[0][0] == approx(1)
    assert jac_wrt_vec2[1][1] == approx(-1)
