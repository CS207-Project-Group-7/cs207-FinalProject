import pytest
import math
import lazydiff.vars as vars
import lazydiff.ops as ops
import numpy as np

def test_sin():
    var1 = vars.Scalar(math.pi)
    var2 = ops.sin(var1)
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == np.array([-1])

def test_cos():
	var1 = vars.Scalar(math.pi)
	var2 = ops.cos(var1)
	assert var2.val == pytest.approx(-1)
	assert var2.grad(var1) == pytest.approx([0])

def test_tan():
	var1 = vars.Scalar(0.)
	var2 = ops.tan(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_asin():
	var1 = vars.Scalar(0.)
	var2 = ops.asin(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_acos():
	var1 = vars.Scalar(0.)
	var2 = ops.acos(var1)
	assert var2.val == pytest.approx(1.570796,abs=1e-2)
	assert var2.grad(var1) == np.array([-1])

def test_atan():
	var1 = vars.Scalar(0.)
	var2 = ops.atan(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_sinh():
	var1 = vars.Scalar(0.)
	var2 = ops.sinh(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_cosh():
	var1 = vars.Scalar(0.)
	var2 = ops.cosh(var1)
	assert var2.val == pytest.approx(1)
	assert var2.grad(var1) == np.array([0])

def test_tanh():
	var1 = vars.Scalar(0.)
	var2 = ops.tanh(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_asinh():
	var1 = vars.Scalar(0.)
	var2 = ops.asinh(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_acosh():
	var1 = vars.Scalar(10.)
	var2 = ops.acosh(var1)
	assert var2.val == pytest.approx(2.993, abs=1e-2)
	assert var2.grad(var1) == np.array([0.10050378])

def test_acosh():
	var1 = vars.Scalar(10.)
	var2 = ops.acosh(var1)
	assert var2.val == pytest.approx(2.99322, abs=1e-3)
	assert var2.grad(var1) == pytest.approx(0.100504, abs=1e-3)

def test_atanh():
	var1 = vars.Scalar(0.)
	var2 = ops.atanh(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_exp():
	var1 = vars.Scalar(0)
	var2 = ops.exp(var1)
	assert var2.val == pytest.approx(1)
	assert var2.grad(var1) == np.array([1])

def test_log():
	var1 = vars.Scalar(1.)
	var2 = ops.log(var1)
	assert var2.val == pytest.approx(0)
	assert var2.grad(var1) == np.array([1])

def test_neg():
	var1 = vars.Scalar(1)
	var2 = ops.neg(var1)
	assert var2.val == pytest.approx(-1)

def test_add():
	var1 = vars.Scalar(1)
	var2 = vars.Scalar(1)
	var3 = ops.add(var1, var2)
	assert var3.val == pytest.approx(2)

def test_sub():
	var1 = vars.Scalar(1)
	var2 = vars.Scalar(1)
	var3 = ops.sub(var1, var2)
	assert var3.val == pytest.approx(0)

def test_mul():
	var1 = vars.Scalar(1)
	var2 = vars.Scalar(1)
	var3 = ops.mul(var1, var2)
	assert var3.val == pytest.approx(1)

def test_div():
	var1 = vars.Scalar(1)
	var2 = vars.Scalar(1)
	var3 = ops.div(var1, var2)
	assert var3.val == pytest.approx(1)

def test_pow():
	var1 = vars.Scalar(1)
	var2 = vars.Scalar(1)
	var3 = ops.pow(var1, var2)
	assert var3.val == pytest.approx(1)
