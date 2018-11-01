import math
import lazydiff.vars as vars

def sin(var):
    """
    Returns variable representing sin applied to the input variable var
    """
    result = vars.Scalar(math.sin(var.val))
    result.parents.append((math.cos(var.val), var))
    return result

def cos(var):
    """
    Returns variable representing cos applied to the input variable var
    """
    result = vars.Scalar(math.cos(var.val))
    result.parents.append((-math.sin(var.val), var))
    return result

def tan(var):
    """
    Returns variable representing tan applied to the input variable var
    """
    result = vars.Scalar(math.tan(var.val))
    result.parents.append((1 / math.cos(var.val) ** 2, var))
    return result

def asin(var):
    """
    Returns variable representing asin applied to the input variable var
    """
    result = vars.Scalar(math.asin(var.val))
    result.parents.append((1 / math.sqrt(1 - var.val ** 2), var))
    return result

def acos(var):
    """
    Returns variable representing acos applied to the input variable var
    """
    result = vars.Scalar(math.atan(var.val))
    result.parents.append((1 / (var.val ** 2 + 1), var))
    return result

def atan(var):
    """
    Returns variable representing atan applied to the input variable var
    """
    result = vars.Scalar(math.asin(var.val))
    result.parents.append((1 / math.sqrt(1 - var.val ** 2), var))
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
    result = vars.Scalar(math.sinh(var.val))
    result.parents.append((math.cosh(var.val), var))
    return result

def cosh(var):
    """
    Returns variable representing cosh applied to the input variable var
    """
    result = vars.Scalar(math.cosh(var.val))
    result.parents.append((math.sinh(var.val), var))
    return result

def tanh(var):
    """
    Returns variable representing tanh applied to the input variable var
    """
    result = vars.Scalar(math.tanh(var.val))
    result.parents.append((1 / (math.cosh(var.val) ** 2), var))
    return result

def asinh(var):
    """
    Returns variable representing asinh applied to the input variable var
    """
    result = vars.Scalar(math.asinh(var.val))
    result.parents.append((1 / math.sqrt(var.val ** 2 + 1), var))
    return result

def acosh(var):
    """
    Returns variable representing acosh applied to the input variable var
    """
    result = vars.Scalar(math.acosh(var.val))
    result.parents.append((1 / math.sqrt(var.val ** 2 - 1), var))
    return result

def atanh(var):
    """
    Returns variable representing atanh applied to the input variable var
    """
    result = vars.Scalar(math.atanh(var.val))
    result.parents.append((1 / (1 - var.val ** 2), var))
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
    result = vars.Scalar(math.exp(var.val))
    result.parents.append((math.exp(var.val), var))
    return result

def log(var, base=math.e):
    """
    Returns variable representing log applied to the input variable var.
    Base of log is optional with default base e
    """
    try:
        result = vars.Scalar(math.log(var.val, base.val))
        result.parents.append((1 / (var.val * math.log(base.val)), var))
        result.parents.append((-math.log(var.val) / (base.val * math.log(base.val) ** 2), base))
    except AttributeError:
        result = vars.Scalar(math.log(var.val, base))
        result.parents.append((1 / (var.val * math.log(base)), var))
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
    return abs(var)



