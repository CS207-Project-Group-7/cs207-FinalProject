import numpy as np
from lazydiff.vars import Scalar, Vector

def vectorize(func):
    def op_wrapper(var, *args):
        try:
            return Vector([func(x, *args) for x in var])
        except TypeError:
            return func(var, *args)
    return op_wrapper

@vectorize
def sin(var):
    """
    Returns variable representing sin applied to the input variable var
    """
    result = Scalar(np.sin(var.val))
    result.parents.append((np.cos(var.val), var))
    return result

@vectorize
def cos(var):
    """
    Returns variable representing cos applied to the input variable var
    """
    result = Scalar(np.cos(var.val))
    result.parents.append((-np.sin(var.val), var))
    return result

@vectorize
def tan(var):
    """
    Returns variable representing tan applied to the input variable var
    """
    result = Scalar(np.tan(var.val))
    result.parents.append((1 / np.cos(var.val) ** 2, var))
    return result

@vectorize
def asin(var):
    """
    Returns variable representing asin applied to the input variable var
    """
    result = Scalar(np.arcsin(var.val))
    result.parents.append((1 / np.sqrt(1 - var.val ** 2), var))
    return result

@vectorize
def acos(var):
    """
    Returns variable representing acos applied to the input variable var
    """
    result = Scalar(np.arccos(var.val))
    result.parents.append((-1 / np.sqrt(1 - var.val ** 2), var))
    return result

@vectorize
def atan(var):
    """
    Returns variable representing atan applied to the input variable var
    """
    result = Scalar(np.arctan(var.val))
    result.parents.append((1 / (var.val ** 2 + 1), var))
    return result

@vectorize
def arcsin(var):
    """
    Wrapper function for asin
    """
    return asin(var)

@vectorize
def arccos(var):
    """
    Wrapper function for asin
    """
    return acos(var)

@vectorize
def arctan(var):
    """
    Wrapper function for atan
    """
    return atan(var)

@vectorize
def sinh(var):
    """
    Returns variable representing sinh applied to the input variable var
    """
    result = Scalar(np.sinh(var.val))
    result.parents.append((np.cosh(var.val), var))
    return result

@vectorize
def cosh(var):
    """
    Returns variable representing cosh applied to the input variable var
    """
    result = Scalar(np.cosh(var.val))
    result.parents.append((np.sinh(var.val), var))
    return result

@vectorize
def tanh(var):
    """
    Returns variable representing tanh applied to the input variable var
    """
    result = Scalar(np.tanh(var.val))
    result.parents.append((1 / (np.cosh(var.val) ** 2), var))
    return result

@vectorize
def asinh(var):
    """
    Returns variable representing asinh applied to the input variable var
    """
    result = Scalar(np.arcsinh(var.val))
    result.parents.append((1 / np.sqrt(var.val ** 2 + 1), var))
    return result

@vectorize
def acosh(var):
    """
    Returns variable representing acosh applied to the input variable var
    """
    result = Scalar(np.arccosh(var.val))
    result.parents.append((1 / np.sqrt(var.val ** 2 - 1), var))
    return result

@vectorize
def atanh(var):
    """
    Returns variable representing atanh applied to the input variable var
    """
    result = Scalar(np.arctanh(var.val))
    result.parents.append((1 / (1 - var.val ** 2), var))
    return result

@vectorize
def arcsinh(var):
    """
    Wrapper function for asinh
    """
    return asinh(var)

@vectorize
def arccosh(var):
    """
    Wrapper function for acosh
    """
    return acosh(var)

@vectorize
def arctanh(var):
    """
    Wrapper function for atanh
    """
    return atanh(var)

@vectorize
def exp(var):
    """
    Returns variable representing exp applied to the input variable var
    """
    result = Scalar(np.exp(var.val))
    result.parents.append((np.exp(var.val), var))
    return result

@vectorize
def log(var, base=np.e):
    """
    Returns variable representing log applied to the input variable var.
    Base of log is optional with default base e
    """
    result = Scalar(np.log(var.val) / np.log(base))
    result.parents.append((1 / (var.val * np.log(base)), var))
    return result

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
