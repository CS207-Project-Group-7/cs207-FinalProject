import math
import numpy as np
import collections
import numbers

np.seterr(all='raise')

def _check_scalar_sequence(args):
    """
    Returns a bool about whether all arguments
    are Var instances
    """
    return np.all([isinstance(arg, Var) for arg in args])

def _get_scalar_sequence(args):
    """
    Returns a bool about whether the arguments
    are Var instances or sequence of Var
    """
    if args == ():
        raise ValueError('Cannot pass in empty argument')
    # checks if arguments are Var
    if _check_scalar_sequence(args):
        return args
    # checks if sequence of Var
    elif len(args) == 1 and _check_scalar_sequence(args[0]):
        return tuple(args[0])
    else:
        raise TypeError("Inputs need to be Var objects or a sequence of Var objects")

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
        self.parents = collections.deque()
        self.children = collections.deque()

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
        args = _get_scalar_sequence(args) 
        result = []
        for i, var in enumerate(args):
            var.forward()
            if var not in self.grad_cache:
                raise ValueError('Variable does not depend on arg {}'.format(i + 1))
            result.append(self.grad_cache[var])
        return result
    
    def forward(self):
        queue = collections.deque([self])
        while queue:
            var = queue.popleft()
            if self not in var.grad_cache:
                grad = np.zeros_like(self.val)
                for parent, factor in var.parents:
                    if self in parent.grad_cache:
                        grad += factor * parent.grad_cache[self]
                var.grad_cache[self] = grad
            for child, _ in var.children:
                queue.append(child)

    def backward(self):
        queue = collections.deque([self])
        while queue:
            var = queue.popleft()
            if var not in self.grad_cache:
                grad = np.zeros_like(var.val)
                for child, factor in var.children:
                    if child in self.grad_cache:
                        grad += factor * self.grad_cache[child]
                self.grad_cache[var] = grad 
            for parent, _ in var.parents:
                queue.append(parent)

    def __getitem__(self, i):
        result = Var(self.val[i])
        factor = np.eye(len(self))[i]
        result.parents.append((self, factor))
        self.children.appendleft((result, factor))
        return result

    def __neg__(self):
        """
        Returns Var object representing negation of a Var object.
        """
        result = Var(-self.val)
        result.parents.append((self, -1.))
        self.children.appendleft((result, -1.))
        return result

    def __abs__(self):
        """
        Returns Var object representing absolute value of a Var object.
        """
        result = Var(np.abs(self.val))
        factor = self.val / np.abs(self.val)
        result.parents.append((self, factor))
        self.children.appendleft((result, factor)) 
        return result

    def __add__(self, other):
        """
        Returns Var object representing addition of two Var objects or
        the addition of a Var object and a Python number.
        """
        if isinstance(other, Var):
            result = Var(self.val + other.val)
            result.parents.append((self, 1.))
            result.parents.append((other, 1.))
            self.children.appendleft((result, 1.))
            other.children.appendleft((result, 1.))
            return result
        elif isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            result = Var(self.val + other)
            result.parents.append((self, 1.))
            self.children.appendleft((result, 1.))
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Var object")

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
            result.parents.append((self, other.val))
            result.parents.append((other, self.val))
            self.children.appendleft((result, self.val))
            other.children.appendleft((result, other.val))
            return result
        elif isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            result = Var(other * self.val)
            result.parents.append((self, other))
            self.children.appendleft((result, other))
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Var object")
    
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
            self_factor = other.val * self.val ** (other.val - 1)
            other_factor = math.log(self.val) * self.val ** other.val
            result.parents.append((self, self_factor))
            result.parents.append((other, other_factor))
            self.children.appendleft((result, self_factor))
            other.children.appendleft((result, other_factor))
            return result
        elif isinstance(other, numbers.Number):
            result = Var(self.val ** other)
            factor = other * self.val ** (other - 1)
            result.parents.append((self, factor))
            self.children.appendleft((result, factor))
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Var object")

    def __rpow__(self, other):
        """
        Returns Var object representing right exponentiation of a Var 
        object with a Python number
        """
        result = Var(other ** self.val)
        factor = math.log(other) * other ** self.val
        result.parents.append((self, factor))
        self.children.appendleft((result, factor))
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