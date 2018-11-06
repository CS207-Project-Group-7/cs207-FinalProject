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
