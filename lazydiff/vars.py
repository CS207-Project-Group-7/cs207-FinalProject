import math

class Var:
    def __init__(self, val):
        self.val = val
        self._grad_cache = {self: 1}
        self.parents = []
    
    def grad(self, *args):
        result = len(args) * [0]
        for i, var in enumerate(args):
            if var not in self._grad_cache:
                self._do_grad(var)
            result[i] = self._grad_cache[var]
        return tuple(result)
    
    def _do_grad(self, var):
        if var not in self._grad_cache:
            self._grad_cache[var] = 0
            for val, parent_var in self.parents:
                parent_var._do_grad(var)
                self._grad_cache[var] += val * parent_var._grad_cache[var]

    def __neg__(self):
        result = Var(-self.val)
        result.parents.append((-1., self))
        return result
    
    def __add__(self, other):
        try:
            result = Var(self.val + other.val)
            result.parents.append((1., other))
        except AttributeError:
            result = Var(self.val + other)
        result.parents.append((1., self))
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            result = Var(self.val * other.val)
            result.parents.append((other.val, self))
            result.parents.append((self.val, other))
        except AttributeError:
            result = Var(other * self.val)
            result.parents.append((other, self))
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        try:
            result = Var(math.pow(self.val, other.val))
            result.parents.append((other.val * math.pow(self.val, other.val - 1), self))
            result.parents.append((math.pow(self.val, other.val) * math.log(self.val), other))
        except AttributeError:
            result = Var(math.pow(self.val, other))
            result.parents.append((other * math.pow(self.val, other - 1), self))
        return result