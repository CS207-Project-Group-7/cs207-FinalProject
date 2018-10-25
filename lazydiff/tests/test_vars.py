import pytest
import lazydiff.vars as vars

def test_init_var():
    var = vars.Var(1)
    assert var.val == 1

def test_add_vars():
    var1 = vars.Var(1)
    var2 = vars.Var(2)
    var3 = var1 + var2
    assert var3.val == 3
    assert var3.grad(var1) == (1,)
    assert var3.grad(var2) == (1,)