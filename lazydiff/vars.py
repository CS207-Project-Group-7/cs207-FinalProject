import math
import numpy as np
import collections
import numbers

np.seterr(all='raise')

class Var:
    """
    A class for lazydiff autograd scalar variables.
    """

    def __init__(self, val, seed=np.array(1.)):
        """
        Initializes Var object with numerical value val.
        """
        self.val = np.array(val, dtype='float')
        self.grad_cache = {self: seed}
        self.parents = {}
        self.children = {}

    def __repr__(self):
        return 'Var({}, seed={})'.format(repr(self.val.tolist()), repr(self.grad_cache[self].tolist()))

    def __len__(self):
        return len(self.val)

    def __hash__(self):
        return id(self)
    
    def grad(self, *args):
        """
        Returns tuple representing gradient with respect to each variable
        provided as arguments.
        """
        if args == ():
            raise ValueError('Cannot pass in empty argument')
        if not np.all([isinstance(arg, Var) for arg in args]):
            raise TypeError("Inputs need to be Var objects or a sequence of Var objects")
        result = []
        for i, var in enumerate(args):
            var.forward()
            if var not in self.grad_cache:
                raise ValueError('Variable does not depend on arg {}'.format(i + 1))
            result.append(self.grad_cache[var])
        return result

    def _forward_visit(self, var, top_sort, seen):
        seen.add(var)
        for child in var.children.keys():
            if child not in seen:
                self._forward_visit(child, top_sort, seen)
        top_sort.appendleft(var)

    def _backward_visit(self, var, top_sort, seen):
        seen.add(var)
        for parent in var.parents.keys():
            if parent not in seen:
                self._backward_visit(parent, top_sort, seen)
        top_sort.appendleft(var)
    
    def forward(self):
        top_sort = collections.deque()
        self._forward_visit(self, top_sort, set())
        while top_sort:
            var = top_sort.popleft()
            if self not in var.grad_cache:
                grad = np.zeros_like(self.val)
                for parent, factor in var.parents.items():
                    if self in parent.grad_cache:
                        grad += factor * parent.grad_cache[self]
                var.grad_cache[self] = grad

    def backward(self):
        top_sort = collections.deque()
        self._backward_visit(self, top_sort, set())
        while top_sort:
            var = top_sort.popleft()
            if var not in self.grad_cache:
                grad = np.zeros_like(var.val)
                for child, factor in var.children.items():
                    if child in self.grad_cache:
                        grad += factor * self.grad_cache[child]
                self.grad_cache[var] = grad 

    def _check_numeric(self, other):
        if isinstance(other, numbers.Number):
            return
        if isinstance(other, np.ndarray) and np.issubdtype(other.dtype, np.number):
            if all(m == n or m == 1 or n == 1 for m, n in zip(self.val.shape[::-1], other.shape[::-1])):
                return
            raise ValueError('Cannot broadcast between shapes')
        raise TypeError("Input needs to be a numeric value, numpy array of numeric values, or Var object")

    def __neg__(self):
        """
        Returns Var object representing negation of a Var object.
        """
        result = Var(-self.val)
        result.parents[self] = self.children[result] = -1.
        return result

    def __abs__(self):
        """
        Returns Var object representing absolute value of a Var object.
        """
        result = Var(abs(self.val))
        result.parents[self] = self.children[result] = self.val / abs(self.val)
        return result

    def __add__(self, other):
        """
        Returns Var object representing addition of two Var objects or
        the addition of a Var object and a Python number.
        """
        if isinstance(other, Var):
            result = Var(self.val + other.val)
            result.parents[self] = self.children[result] = 1.
            result.parents[other] = other.children[result] = 1.
            return result
        self._check_numeric(other)
        result = Var(self.val + other)
        result.parents[self] = self.children[result] = 1.
        return result

    def __radd__(self, other):
        """
        Returns Var object representing right addition of a Var object
        with a Python number.
        """
        return self + other

    def __sub__(self, other):
        """
        Returns Var object representing subtraction of two Var objects or
        the subtraction of a Var object and a Python number
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Returns Var object representing right subtraction of a Var object
        with a Python number
        """
        return -self + other

    def __mul__(self, other):
        """
        Returns Var object representing multiplication of two Var objects 
        or the multiplication of a Var object and a Python number
        """
        if isinstance(other, Var):
            result = Var(self.val * other.val)
            result.parents[self] = self.children[result] = other.val
            result.parents[other] = other.children[result] = self.val
            return result
        self._check_numeric(other)
        result = Var(other * self.val)
        result.parents[self] = self.children[result] = other
        return result
    
    def __rmul__(self, other):
        """
        Returns Var object representing right multiplication of a Var 
        object with a Python number
        """
        return self * other

    def __truediv__(self, other):
        """
        Returns Var object representing division of two Var objects or 
        the division of a Var object and a Python number
        """
        return self * (other ** -1)

    def __rtruediv__(self, other):
        """
        Returns Var object representing right division of a Var object
        with a Python number
        """
        return (self ** -1) * other

    def __pow__(self, other):
        """
        Returns Var object representing exponentiation of two Var objects 
        or the exponentiation of a Var object and a Python number
        """
        if isinstance(other, Var):
            result = Var(self.val ** other.val)
            result.parents[self] = self.children[result] = other.val * self.val ** (other.val - 1)
            result.parents[other] = other.children[result] = math.log(self.val) * self.val ** other.val
            return result
        self._check_numeric(other)
        result = Var(self.val ** other)
        result.parents[self] = self.children[result] = other * self.val ** (other - 1)
        return result

    def __rpow__(self, other):
        """
        Returns Var object representing right exponentiation of a Var 
        object with a Python number
        """
        self._check_numeric(other)
        result = Var(other ** self.val)
        result.parents[self] = self.children[result] = math.log(other) * other ** self.val
        return result

    def __eq__(self, other):
        return isinstance(other, Var) and np.all(self.val == other.val)
    
    def __ne__(self, other):
        return isinstance(other, Var) and np.all(self.val != other.val)

    def __lt__(self, other):
        return isinstance(other, Var) and np.all(self.val < other.val)

    def __gt__(self, other):
        return isinstance(other, Var) and np.all(self.val > other.val)

    def __le_(self, other):
        return isinstance(other, Var) and np.all(self.val <= other.val)

    def __ge__(self, other):
        return isinstance(other, Var) and np.all(self.val >= other.val)

    def _in_place_error():
        """
        Raises error for in-place operations
        """
        raise TypeError("In-place operations are not supported for lazydiff variables.")

    def __iadd__(self):
        _in_place_error()

    def __isub__(self):
        _in_place_error()

    def __imul__(self):
        _in_place_error()

    def __itruediv__(self):
        _in_place_error()

    def __ipow__(self):
        _in_place_error()