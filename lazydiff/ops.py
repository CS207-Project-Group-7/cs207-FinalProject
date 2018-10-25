import math
import lazydiff.vars as vars

def sin(self):
    result = vars.Var(math.sin(self.val))
    result.parents.append((math.cos(self.val), self))
    return result

def cos(self):
    result = vars.Var(math.cos(self.val))
    result.parents.append((-math.sin(self.val), self))
    return result

def exp(self):
    result = vars.Var(math.exp(self.val))
    result.parents.append((math.exp(self.val), self))
    return result