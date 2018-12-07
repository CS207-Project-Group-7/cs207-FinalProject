import numpy as np
from lazydiff.vars import Var

def sin(var):
    """
    Returns variable representing sin applied to the input variable var
    """
    result = Var(np.sin(var.val))
    factor = np.cos(var.val)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor)) 
    return result

def cos(var):
    """
    Returns variable representing cos applied to the input variable var
    """
    result = Var(np.cos(var.val))
    factor = -np.sin(var.val)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def tan(var):
    """
    Returns variable representing tan applied to the input variable var
    """
    result = Var(np.tan(var.val))
    factor = 1 / np.cos(var.val) ** 2
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def asin(var):
    """
    Returns variable representing asin applied to the input variable var
    """
    result = Var(np.arcsin(var.val))
    factor = 1 / np.sqrt(1 - var.val ** 2)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def acos(var):
    """
    Returns variable representing acos applied to the input variable var
    """
    result = Var(np.arccos(var.val))
    factor = -1 / np.sqrt(1 - var.val ** 2)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def atan(var):
    """
    Returns variable representing atan applied to the input variable var
    """
    result = Var(np.arctan(var.val))
    factor = 1 / (var.val ** 2 + 1)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
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
    factor = np.cosh(var.val)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def cosh(var):
    """
    Returns variable representing cosh applied to the input variable var
    """
    result = Var(np.cosh(var.val))
    factor = np.sinh(var.val)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def tanh(var):
    """
    Returns variable representing tanh applied to the input variable var
    """
    result = Var(np.tanh(var.val))
    factor = 1 / (np.cosh(var.val) ** 2)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def asinh(var):
    """
    Returns variable representing asinh applied to the input variable var
    """
    result = Var(np.arcsinh(var.val))
    factor = 1 / np.sqrt(var.val ** 2 + 1)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def acosh(var):
    """
    Returns variable representing acosh applied to the input variable var
    """
    result = Var(np.arccosh(var.val))
    factor = 1 / np.sqrt(var.val ** 2 - 1)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def atanh(var):
    """
    Returns variable representing atanh applied to the input variable var
    """
    result = Var(np.arctanh(var.val))
    factor = 1 / (1 - var.val ** 2)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
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
    factor = np.exp(var.val)
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def log(var, base=np.e):
    """
    Returns variable representing log applied to the input variable var.
    Base of log is optional with default base e
    """
    result = Var(np.log(var.val) / np.log(base))
    factor = 1 / (var.val * np.log(base))
    result.parents.append((var, factor))
    var.children.appendleft((result, factor))
    return result


def logistic(var):
    return 1 / (1 + exp(-var))


def sqrt(var):
    return var ** 0.5

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
