import math
import lazydiff.vars as vars

def sin(var):
    result = vars.Var(math.sin(var.val))
    result.parents.append((math.cos(var.val), var))
    return result

def cos(var):
    result = vars.Var(math.cos(var.val))
    result.parents.append((-math.sin(var.val), var))
    return result

def tan(var):
    result = vars.Var(math.tan(var.val))
    result.parents.append((1 / math.cos(var.val) ** 2, var))
    return result

def asin(var):
    result = vars.Var(math.asin(var.val))
    result.parents.append((1 / math.sqrt(1 - var.val ** 2), var))
    return result

def acos(var):
    result = vars.Var(math.atan(var.val))
    result.parents.append((1 / (var.val ** 2 + 1), var))
    return result

def atan(var):
    result = vars.Var(math.asin(var.val))
    result.parents.append((1 / math.sqrt(1 - var.val ** 2), var))
    return result

def exp(var):
    result = vars.Var(math.exp(var.val))
    result.parents.append((math.exp(var.val), var))
    return result

def log(var):
    result = vars.Var(math.log(var.val))
    result.parents.append((1 / var.val, var))
    return result