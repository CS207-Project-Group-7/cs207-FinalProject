import pytest
import math
import lazydiff.vars as vars
import lazydiff.ops as ops

def test_sin():
    var1 = vars.Var(math.pi)
    var2 = ops.sin(var1)
    assert var2.val == pytest.approx(0)
    assert var2.grad(var1) == (-1,)