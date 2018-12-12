import numpy as np
from lazydiff.vars import Var

def sin(var):
    """
    Returns variable representing sin applied to the input variable var
    """
    result = Var(np.sin(var.val))
    result.parents[var] = var.children[result] = np.cos(var.val)
    return result

def cos(var):
    """
    Returns variable representing cos applied to the input variable var
    """
    result = Var(np.cos(var.val))
    result.parents[var] = var.children[result] = -np.sin(var.val)
    return result

def tan(var):
    """
    Returns variable representing tan applied to the input variable var
    """
    result = Var(np.tan(var.val))
    result.parents[var] = var.children[result] = 1 / np.cos(var.val) ** 2
    return result

def asin(var):
    """
    Returns variable representing asin applied to the input variable var
    """
    result = Var(np.arcsin(var.val))
    result.parents[var] = var.children[result] = 1 / np.sqrt(1 - var.val ** 2)
    return result

def acos(var):
    """
    Returns variable representing acos applied to the input variable var
    """
    result = Var(np.arccos(var.val))
    result.parents[var] = var.children[result] = -1 / np.sqrt(1 - var.val ** 2)
    return result

def atan(var):
    """
    Returns variable representing atan applied to the input variable var
    """
    result = Var(np.arctan(var.val))
    result.parents[var] = var.children[result] = 1 / (var.val ** 2 + 1)
    return result

def arcsin(var):
    """
    Wrapper function for asin
    """
    return asin(var)

def arccos(var):
    """
    Wrapper function for asin
    """
    return acos(var)

def arctan(var):
    """
    Wrapper function for atan
    """
    return atan(var)

def sinh(var):
    """
    Returns variable representing sinh applied to the input variable var
    """
    result = Var(np.sinh(var.val))
    result.parents[var] = var.children[result] = np.cosh(var.val)
    return result

def cosh(var):
    """
    Returns variable representing cosh applied to the input variable var
    """
    result = Var(np.cosh(var.val))
    result.parents[var] = var.children[result] = np.sinh(var.val)
    return result

def tanh(var):
    """
    Returns variable representing tanh applied to the input variable var
    """
    result = Var(np.tanh(var.val))
    result.parents[var] = var.children[result] = 1 / (np.cosh(var.val) ** 2)
    return result

def asinh(var):
    """
    Returns variable representing asinh applied to the input variable var
    """
    result = Var(np.arcsinh(var.val))
    result.parents[var] = var.children[result] = 1 / np.sqrt(var.val ** 2 + 1)
    return result

def acosh(var):
    """
    Returns variable representing acosh applied to the input variable var
    """
    result = Var(np.arccosh(var.val))
    result.parents[var] = var.children[result] = 1 / np.sqrt(var.val ** 2 - 1)
    return result

def atanh(var):
    """
    Returns variable representing atanh applied to the input variable var
    """
    result = Var(np.arctanh(var.val))
    result.parents[var] = var.children[result] = 1 / (1 - var.val ** 2)
    return result

def arcsinh(var):
    """
    Wrapper function for asinh
    """
    return asinh(var)

def arccosh(var):
    """
    Wrapper function for acosh
    """
    return acosh(var)

def arctanh(var):
    """
    Wrapper function for atanh
    """
    return atanh(var)

def exp(var):
    """
    Returns variable representing exp applied to the input variable var
    """
    result = Var(np.exp(var.val))
    result.parents[var] = var.children[result] = np.exp(var.val)
    return result

def log(var, base=np.e):
    """
    Returns variable representing log applied to the input variable var.
    Base of log is optional with default base e
    """
    result = Var(np.log(var.val) / np.log(base))
    result.parents[var] = var.children[result] = 1 / (var.val * np.log(base))
    return result

def logistic(var):
    """
    Returns variable representing sigmoid applied to input variable var
    """
    return (1 + exp(-var)) ** -1

def sqrt(var):
    """
    Returns variable representing square root applied to input variable var
    """
    return var ** 0.5

def sum(var):
    """
    Returns variable representing the sum of the components of input variable var
    """
    result = Var(np.sum(var.val))
    result.parents[var] = var.children[result] = np.ones_like(var.val)
    return result    

def norm(var, p=1):
    """
    Returns variable representing L-p norm of input variable var
    """
    return sum(abs(var) ** p) ** (1 / p)

def neg(var):
    """
    Wrapper function for __neg__
    """
    return -var

def add(var1, var2):
    """
    Wrapper function for __add__ and __radd__
    """
    return var1 + var2

def sub(var1, var2):
    """
    Wrapper function for __sub__ and __rsub__
    """
    return var1 - var2

def mul(var1, var2):
    """
    Wrapper function for __mul__ and __rmul__
    """
    return var1 * var2

def div(var1, var2):
    """
    Wrapper function for __truediv__ and __rtruediv__
    """
    return var1 / var2

def pow(var1, var2):
    """
    Wrapper function for __pow__
    """
    return var1 ** var2

def abs(var):
    """
    Wrapper function for __abs__
    """
    return var.__abs__()
    