import math
import numpy as np
import collections
import numbers

def _check_scalar_sequence(args):
    """
    Returns a bool about whether all arguments
    are Scalar instances
    """
    return np.all([isinstance(arg, Scalar) for arg in args])

def _get_scalar_sequence(args):
    """
    Returns a bool about whether the arguments
    are Scalar instances or sequence of Scalar
    """
    if args == ():
        raise ValueError('Cannot pass in empty argument')
    # checks if arguments are Scalar
    if _check_scalar_sequence(args):
        return args
    # checks if sequence of Scalar
    elif len(args) == 1 and _check_scalar_sequence(args[0]):
        return tuple(args[0])
    else:
        raise TypeError("Inputs need to be Scalar objects or a sequence of Scalar objects")

def _in_place_error(*args):
    """
    Raises error for in-place operations
    """
    raise TypeError("In-place operations are not supported for lazydiff variables.")

def ban_in_place(cls):
    """
    Decorator that bans in-place operations
    """
    for op in ['add', 'sub', 'mul', 'truediv', 'pow']:
        setattr(cls, '__i{}__'.format(op), _in_place_error)
    return cls

@ban_in_place
class Scalar:
    """
    A class for lazydiff autograd scalar variables.
    """

    def __init__(self, val, seed=[1.]):
        """
        Initializes Scalar object with numerical value val.
        """
        self.val = np.atleast_1d(np.array(val, dtype='float'))
        self.grad_cache = dict()
        self.grad_cache[self] = np.array(seed)
        self.parents = {}
        self.children = {}

    def __repr__(self):
        return 'Scalar({})'.format(repr(self.val.tolist()))

    def __hash__(self):
        return id(self)
    
    def grad(self, *args):
        """
        Returns tuple representing gradient with respect to each variable
        provided as arguments.
        """
        args = _get_scalar_sequence(args) 
        result = np.zeros((len(self.val),len(args)))
        for i, var in enumerate(args):
            if var not in self.grad_cache:
                raise ValueError('Variable does not depend on arg {}'.format(i + 1))
            result[:,i] = self.grad_cache[var]
        return result
    
    def forward(self):
        queue = collections.deque([self])
        while queue:
            var = queue.popleft()
            if self not in var.grad_cache:
                grad = np.zeros_like(var.val)
                for parent, val in var.parents.items():
                    if self in parent.grad_cache:
                        grad += val * parent.grad_cache[self]
                var.grad_cache[self] = grad
            for child in var.children:
                queue.append(child)

    def backward(self):
        queue = collections.deque([self])
        while queue:
            var = queue.popleft()
            if var not in self.grad_cache:
                grad = np.zeros_like(var.val)
                for child, val in var.children.items():
                    if child in self.grad_cache:
                        grad += val * self.grad_cache[child]
                self.grad_cache[var] = grad 
            for parent in var.parents:
                queue.append(parent)

    def __neg__(self):
        """
        Returns Scalar object representing negation of a Scalar object.
        """
        result = Scalar(-self.val)
        result.parents[self] = self.children[result] = -1.
        return result

    def __abs__(self):
        """
        Returns Scalar object representing absolute value of a Scalar object.
        """
        result = Scalar(np.abs(self.val))
        result.parents[self] = self.children[result] = self.val / np.abs(self.val)
        return result

    def __add__(self, other):
        """
        Returns Scalar object representing addition of two Scalar objects or
        the addition of a Scalar object and a Python number.
        """
        if isinstance(other, Scalar):
            result = Scalar(self.val + other.val)
            result.parents[self] = self.children[result] = 1.
            result.parents[other] = other.children[result] = 1.
            return result
        elif isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            result = Scalar(self.val + other)
            result.parents[self] = self.children[result] = 1.
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Scalar object")

    def __radd__(self, other):
        """
        Returns Scalar object representing right addition of a Scalar object
        with a Python number.
        """
        return self + other

    def __sub__(self, other):
        """
        Returns Scalar object representing subtraction of two Scalar objects or
        the subtraction of a Scalar object and a Python number
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Returns Scalar object representing right subtraction of a Scalar object
        with a Python number
        """
        return -self + other

    def __mul__(self, other):
        """
        Returns Scalar object representing multiplication of two Scalar objects 
        or the multiplication of a Scalar object and a Python number
        """
        if isinstance(other, Scalar):
            result = Scalar(self.val * other.val)
            result.parents[self] = self.children[result] = other.val
            result.parents[other] = other.children[result] = self.val
            return result
        elif isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            result = Scalar(other * self.val)
            result.parents[self] = self.children[result] = other
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Scalar object")
    
    def __rmul__(self, other):
        """
        Returns Scalar object representing right multiplication of a Scalar 
        object with a Python number
        """
        return self * other

    def __truediv__(self, other):
        """
        Returns Scalar object representing division of two Scalar objects or 
        the division of a Scalar object and a Python number
        """
        return self * (other ** -1)

    def __rtruediv__(self, other):
        """
        Returns Scalar object representing right division of a Scalar object
        with a Python number
        """
        return (self ** -1) * other

    def __pow__(self, other):
        """
        Returns Scalar object representing exponentiation of two Scalar objects 
        or the exponentiation of a Scalar object and a Python number
        """
        if isinstance(other, Scalar):
            result = Scalar(self.val ** other.val)
            result.parents[self] = self.children[result] = other.val * self.val ** (other.val - 1)
            result.parents[other] = other.children[result] = math.log(self.val) * self.val ** other.val
            return result
        elif isinstance(other, numbers.Number):
            result = Scalar(self.val ** other)
            result.parents[self] = self.children[result] = other * self.val ** (other - 1)
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Scalar object")

    def __rpow__(self, other):
        """
        Returns Scalar object representing right exponentiation of a Scalar 
        object with a Python number
        """
        result = Scalar(other ** self.val)
        result.parents[self] = self.children[result] = math.log(other) * other ** self.val
        return result

    def __eq__(self, other):
        return isinstance(other, Scalar) and np.all(self.val == other.val)
    
    def __ne__(self, other):
        return isinstance(other, Scalar) and np.all(self.val != other.val)

    def __lt__(self, other):
        return isinstance(other, Scalar) and np.all(self.val < other.val)

    def __gt__(self, other):
        return isinstance(other, Scalar) and np.all(self.val > other.val)

    def __le_(self, other):
        return isinstance(other, Scalar) and np.all(self.val <= other.val)

    def __ge__(self, other):
        return isinstance(other, Scalar) and np.all(self.val >= other.val)

@ban_in_place
class Vector:
    """
    A class for lazydiff autograd vector variables.
    """

    def __init__(self, *args):
        """
        Initializes Scalar object with arguments of Scalar objects
        or a sequence of Scalar objects args
        """
        self._components = _get_scalar_sequence(args) 
        self.val = np.array([component.val for component in self._components])

    def __repr__(self):
        return 'Vector(%s)' % repr(self.val)

    def grad(self, *args):
        """
        Returns numpy array representing Jacobian
        with respect to each variable provided as arguments args
        """
        return np.array([component.grad(*args) for component in self._components])

    def forward(self):
        [component.forward() for component in self._components]

    def backward(self):
        [component.backward() for component in self._components]      

    def __getitem__(self, ind):
        """
        Returns a Scalar object in the given index
        """
        if ind not in range(len(self)):
            raise IndexError
        return self._components[ind]

    def __len__(self):
        """
        Returns the number of Scalar objects the vector holds
        """
        return len(self._components)

    def _check_broadcast(self, other):
        """
        Returns a bool about whether operands are of matching length
        """
        if (len(self) != len(other)):
            raise ValueError("Operands could not be broadcast together with lengths {} and {}".format(len(self),len(other)))

    def _unop_wrapper(self, op):
        """
        Returns a Vector instance unwrapping unitary operation
        """
        return Vector([op(component) for component in self._components])

    def _binop_wrapper(self, other, op):
        """
        Returns a Vector instance unwrapping binary operation
        """
        if isinstance(other, numbers.Number) or isinstance(other, Scalar):
            return Vector([op(component, other) for component in self._components])
        elif isinstance(other, Vector):
            self._check_broadcast(other)
            return Vector([op(comp1, comp2) for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("Input needs to be a numeric value, Scalar object, or Vector object")

    def _rop_wrapper(self, other, op):
        """
        Returns a Vector instance unwrapping reverse binary operations
        """
        if isinstance(other, numbers.Number) or isinstance(other, Scalar):
            return Vector([op(component, other) for component in self._components])
        else:
            raise TypeError("Input needs to be a numeric value or Scalar object")

    def _comp_wrapper(self, other, op):
        if isinstance(other, Vector):
            self._check_broadcast(other)
            return np.array([op(comp1, comp2) for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("Input needs to be a Vector object")

    def __neg__(self):
        """
        Returns a Vector instance with the negation of the elements
        """
        return self._unop_wrapper(Scalar.__neg__)

    def __abs__(self):
        """
        Returns a Vector instance with the absolute of the elements
        """
        return self._unop_wrapper(Scalar.__abs__)
    
    def __add__(self, other):
        """
        Returns a Vector instance representing addition of 
        two Vector instances or the addition of a Scalar instance
        and a number by broadcasting
        """
        return self._binop_wrapper(other, Scalar.__add__)

    def __radd__(self, other):
        """
        Returns a Vector instance representing right addition of 
        two Vector instances or the right addition of a Scalar instance
        and a number by broadcasting
        """
        return self._rop_wrapper(other, Scalar.__radd__)

    def __sub__(self, other):
        """
        Returns a Vector instance representing subtraction of 
        two Vector instances or the subtraction of a Scalar instance
        and a number by broadcasting
        """
        return self._binop_wrapper(other, Scalar.__sub__)

    def __rsub__(self, other):
        """
        Returns a Vector instance representing right subtraction of 
        two Vector instances or the right subtraction of a Scalar instance
        and a number by broadcasting
        """
        return self._rop_wrapper(other, Scalar.__rsub__)

    def __mul__(self, other):
        """
        Returns a Vector instance representing multiplication of 
        two Vector instances or the multiplication of a Scalar instance
        and a number by broadcasting
        """
        return self._binop_wrapper(other, Scalar.__mul__)
    
    def __rmul__(self, other):
        """
        Returns a Vector instance representing right multiplication of 
        two Vector instances or the right multiplication of a Scalar instance
        and a number by broadcasting
        """
        return self._rop_wrapper(other, Scalar.__rmul__)

    def __truediv__(self, other):
        """
        Returns a Vector instance representing division of 
        two Vector instances or the division of a Scalar instance
        and a number by broadcasting
        """
        return self._binop_wrapper(other, Scalar.__truediv__)

    def __rtruediv__(self, other):
        """
        Returns a Vector instance representing right division of 
        two Vector instances or the right division of a Scalar instance
        and a number by broadcasting
        """
        return self._rop_wrapper(other, Scalar.__rtruediv__)

    def __pow__(self, other):
        """
        Returns Vector instance representing exponentiation of 
        twi Vector instances or the exponentiation of a Scalar instance
        and a number by broadcasting
        """
        return self._binop_wrapper(other, Scalar.__pow__)

    def __rpow__(self, other):
        """
        Returns Vector instance representing right exponentiation of 
        twi Vector instances or the right exponentiation of a Scalar instance
        and a number by broadcasting
        """
        return self._rop_wrapper(other, Scalar.__rpow__)

    def __eq__(self, other):
        return self._comp_wrapper(other, Scalar.__eq__)
    
    def __ne__(self, other):
        return self._comp_wrapper(other, Scalar.__ne__)

    def __lt__(self, other):
        return self._comp_wrapper(other, Scalar.__lt__)

    def __gt__(self, other):
        return self._comp_wrapper(other, Scalar.__gt__)

    def __le_(self, other):
        return self._comp_wrapper(other, Scalar.__le__)

    def __ge__(self, other):
        return self._comp_wrapper(other, Scalar.__ge__)
