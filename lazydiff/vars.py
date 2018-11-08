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

    def __init__(self, val):
        """
        Initializes Scalar object with numerical value val.
        """
        self.val = val
        self._grad_cache = {self: 1}
        self.parents = []
    
    def grad(self, *args):
        """
        Returns tuple representing gradient with respect to each variable
        provided as arguments.
        """
        args = _get_scalar_sequence(args) 
        result = np.zeros(len(args))
        for i, var in enumerate(args):
            self._compute_grad(var)
            result[i] = self._grad_cache[var]
        return result
    
    def _compute_grad(self, var):
        """
        Internal method for computing gradient with respect to variable var
        and storing the result in the gradient cache dictionary.
        """
        if var not in self._grad_cache:
            grad = 0
            for val, parent_var in self.parents:
                parent_var._compute_grad(var)
                grad += val * parent_var._grad_cache[var]
            self._grad_cache[var] = grad

    def __neg__(self):
        """
        Returns Scalar object representing negation of a Scalar object.
        """
        result = Scalar(-self.val)
        result.parents.append((-1., self))
        return result

    def __abs__(self):
        """
        Returns Scalar object representing absolute value of a Scalar object.
        """
        result = Scalar(abs(self.val))
        result.parents.append((self.val / abs(self.val), self))
        return result

    def __add__(self, other):
        """
        Returns Scalar object representing addition of two Scalar objects or
        the addition of a Scalar object and a Python number.
        """
        if isinstance(other, Scalar):
            result = Scalar(self.val + other.val)
            result.parents.append((1., self))
            result.parents.append((1., other))
            return result
        elif isinstance(other, numbers.Number):
            result = Scalar(self.val + other)
            result.parents.append((1., self))
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Vector object")

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
            result.parents.append((other.val, self))
            result.parents.append((self.val, other))
            return result
        elif isinstance(other, numbers.Number):
            result = Scalar(other * self.val)
            result.parents.append((other, self))
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Vector object")
    
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
            result.parents.append((other.val * self.val ** (other.val - 1), self))
            result.parents.append((math.log(self.val) * self.val ** other.val, other))
            return result
        elif isinstance(other, numbers.Number):
            result = Scalar(self.val ** other)
            result.parents.append((other * self.val ** (other - 1), self))
            return result
        else:
            raise TypeError("Input needs to be a numeric value or Vector object")

    def __rpow__(self, other):
        """
        Returns Scalar object representing right exponentiation of a Scalar 
        object with a Python number
        """
        result = Scalar(other ** self.val)
        result.parents.append((math.log(other) * other ** self.val, self))
        return result

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
        self.val = tuple([component.val for component in self._components])

    def grad(self, *args):
        """
        Returns numpy array representing Jacobian
        with respect to each variable provided as arguments args
        """
        return np.array([component.grad(*args) for component in self._components])      

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
        return Vector([op(component, other) for component in self._components])

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
