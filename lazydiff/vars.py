import math

class Scalar:
    def __init__(self, val):
        self.val = val
        self._grad_cache = {self: 1}
        self.parents = []
    
    def grad(self, *args):
        result = len(args) * [0]
        for i, var in enumerate(args):
            if var not in self._grad_cache:
                self._compute_grad(var)
            result[i] = self._grad_cache[var]
        return tuple(result)
    
    def _compute_grad(self, var):
        if var not in self._grad_cache:
            grad = 0
            for val, parent_var in self.parents:
                parent_var._compute_grad(var)
                grad += val * parent_var._grad_cache[var]
            self._grad_cache[var] = grad

    def __neg__(self):
        result = Scalar(-self.val)
        result.parents.append((-1., self))
        return result
    
    def __add__(self, other):
        try:
            result = Scalar(self.val + other.val)
            result.parents.append((1., other))
        except AttributeError:
            result = Scalar(self.val + other)
        result.parents.append((1., self))
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        try:
            result = Scalar(self.val * other.val)
            result.parents.append((other.val, self))
            result.parents.append((self.val, other))
        except AttributeError:
            result = Scalar(other * self.val)
            result.parents.append((other, self))
        return result
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other):
        return self.__pow__(-1).__mul__(other)

    def __pow__(self, other):
        try:
            result = Scalar(math.pow(self.val, other.val))
            result.parents.append((other.val * math.pow(self.val, other.val - 1), self))
            result.parents.append((math.pow(self.val, other.val) * math.log(self.val), other))
        except AttributeError:
            result = Scalar(math.pow(self.val, other))
            result.parents.append((other * math.pow(self.val, other - 1), self))
        return result

class Vector:
    def __init__(self, *args):
        self._components = args
        self.val = tuple([component.val for component in self._components])

    def grad(self, *args):
        return tuple([component.grad(*args) for component in self._components])

    def __getitem__(self, ind):
        if ind not in range(self.__len__()):
            raise IndexError
        return self._components[ind]

    def __len__(self):
        return len(self._component)