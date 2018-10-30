import math
import lazydiff.vars as vars

def sin(var):
    result = vars.Scalar(math.sin(var.val))
    result.parents.append((math.cos(var.val), var))
    return result

def cos(var):
    result = vars.Scalar(math.cos(var.val))
    result.parents.append((-math.sin(var.val), var))
    return result

def tan(var):
    result = vars.Scalar(math.tan(var.val))
    result.parents.append((1 / math.cos(var.val) ** 2, var))
    return result

def asin(var):
    result = vars.Scalar(math.asin(var.val))
    result.parents.append((1 / math.sqrt(1 - var.val ** 2), var))
    return result

def acos(var):
    result = vars.Scalar(math.atan(var.val))
    result.parents.append((1 / (var.val ** 2 + 1), var))
    return result

def atan(var):
    result = vars.Scalar(math.asin(var.val))
    result.parents.append((1 / math.sqrt(1 - var.val ** 2), var))
    return result

def arcsin(var):
    return asin(var)

def arccos(var):
    return acos(var)

def arctan(var):
    return atan(var)

def sinh(var):
    result = vars.Scalar(math.sinh(var.val))
    result.parents.append((math.cosh(var.val), var))
    return result

def cosh(var):
    result = vars.Scalar(math.cosh(var.val))
    result.parents.append((math.sinh(var.val), var))
    return result

def tanh(var):
    result = vars.Scalar(math.tanh(var.val))
    result.parents.append((1 / (math.cosh(var.val)**2), var))
    return result

def asinh(var):
    result = vars.Scalar(math.asinh(var.val))
    result.parents.append((1 / math.sqrt(var.val ** 2 + 1), var))
    return result

def acosh(var):
    result = vars.Scalar(math.acosh(var.val))
    result.parents.append((1 / (math.sqrt(var.val-1) * math.sqrt(var.val+1)), var))
    return result

def atanh(var):
    result = vars.Scalar(math.atanh(var.val))
    result.parents.append((1/ (1 - var.val ** 2), var))
    return result

def arcsinh(var):
    return asinh(var)

def arccosh(var):
    return acosh(var)

def arctanh(var):
    return atanh(var)

def exp(var):
    result = vars.Scalar(math.exp(var.val))
    result.parents.append((math.exp(var.val), var))
    return result

def log(var, base = math.e):
    result = vars.Scalar(math.log(var.val, base))
    result.parents.append((1 / (var.val * math.log(base)), var))
    return result

def power(var, exponent):
    return var**exponent



