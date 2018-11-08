import pytest
import math
from lazydiff.vars import Vector, Scalar
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

def test_vec_abs():
    vec = Vector([Scalar(-i) for i in range(1,4)])
    vec2 = ops.abs(vec)
    assert vec2.val == approx((1,2,3))
    assert np.all(vec2.grad(vec)[i][i] == approx(-1) for i in range(3))

