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
        self.grad_val = {self: seed}
        self.parents = {}
        self.children = {}

    def __repr__(self):
        """
        Returns string representation of Var object
        """
        return 'Var({}, seed={})'.format(repr(self.val.tolist()), repr(self.grad_val[self].tolist()))

    def __hash__(self):
        """
        Computes hash of Var object
        """
        return id(self)

    def grad(self, var):
        """
        Returns numpy array representing gradient with respect to variable var.
        Raises error if self does not depend on variable var.
        """
        if not isinstance(var, Var):
            raise TypeError('Inputs needs to be Var object.')
        if var not in self.grad_val:
            raise ValueError('Variable does not depend on input var. Make sure you have run forward/backward.')
        return self.grad_val[var]

    def _forward_visit(self, var, top_sort, seen):
        """
        Auxiliary method for getting topological order for visiting nodes in 
        forward. Takes in Var object var to start from, deque top_sort that will
        hold the resulting topological order, and set seen containing previously
        seen nodes in the course of creating the topological order.
        """
        seen.add(var)
        for child in var.children.keys():
            if child not in seen:
                self._forward_visit(child, top_sort, seen)
        top_sort.appendleft(var)
    
    def forward(self):
        """
        Propagates gradients forward from this variable.
        Before making any call var.grad(self), where var is a variable that
        depends on self, either need to run self.forward() or var.backward().
        """
        top_sort = collections.deque()
        self._forward_visit(self, top_sort, set())
        for var in top_sort:
            if not var is self:
                grad = np.array(0.)
                for parent, factor in var.parents.items():
                    if self in parent.grad_val:
                        grad = grad + factor * parent.grad_val[self]
                var.grad_val[self] = grad

    def _backward_visit(self, var, top_sort, seen):
        """
        Auxiliary method for getting topological order for visiting nodes in 
        backward. Takes in Var object var to start from, deque top_sort that will
        hold the resulting topological order, and set seen containing previously
        seen nodes in the course of creating the topological order.
        """
        seen.add(var)
        for parent in var.parents.keys():
            if parent not in seen:
                self._backward_visit(parent, top_sort, seen)
        top_sort.appendleft(var)

    def backward(self):
        """
        Propagates gradients backward from this variable.
        Before making any call self.grad(var), where var is a variable on which
        self depends, either need to run self.backward() or var.forward().
        """
        top_sort = collections.deque()
        self._backward_visit(self, top_sort, set())
        for var in top_sort:
            if not var is self:
                grad = np.array(0.) 
                for child, factor in var.children.items():
                    if child in self.grad_val:
                        grad = grad + factor * self.grad_val[child]
                self.grad_val[var] = grad 

    def _check_numeric(self, other):
        """
        Checks if given object is of numeric type or is a numpy array of numeric types
        """
        if not isinstance(other, numbers.Number) and not (isinstance(other, np.ndarray) 
            and np.issubdtype(other.dtype, np.number)):
            raise TypeError("Input needs to be numeric value, numpy array of numeric values, or Var object")

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
            result.parents[other] = other.children[result] = np.log(self.val) * self.val ** other.val
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
        result.parents[self] = self.children[result] = np.log(other) * other ** self.val
        return result

    def _in_place_error(self):
        """
        Raises error for in-place operations
        """
        raise TypeError("In-place operations are not supported for lazydiff variables.")

    def __iadd__(self, other):
        """
        Ban in-place addition for Var objects
        """
        self._in_place_error()

    def __isub__(self, other):
        """
        Ban in-place subtraction for Var objects.
        """
        self._in_place_error()

    def __imul__(self, other):
        """
        Ban in-place multiplication for Var objects.
        """
        self._in_place_error()

    def __itruediv__(self, other):
        """
        Ban in-place division for Var objects.
        """
        self._in_place_error()

    def __ipow__(self, other):
        """
        Ban in-place exponentiation for Var objects.
        """
        self._in_place_error()

    def _comparison(self, other, op):
        """
        Performs comparison for given comparison op between Var object and another object
        """
        if isinstance(other, Var):
            return op(self.val, other.val)
        return False

    def __eq__(self, other):
        """
        Checks if Var object is equal to another object. 
        If other object is Var object, returns result of numpy comparison of their values.
        """
        return self._comparison(other, np.ndarray.__eq__)
    
    def __ne__(self, other):
        """
        Checks if Var object is not equal to another object.
        If other object is Var object, returns result of numpy comparison of their values.
        """
        return ~(self == other)

    def __lt__(self, other):
        """
        Checks if Var object is less than another object.
        If other object is Var object, returns result of numpy comparison of their values.
        """
        return self._comparison(other, np.ndarray.__lt__)

    def __gt__(self, other):
        """
        Checks if Var object is greater than another object.
        If other object is Var object, returns result of numpy comparison of their values.
        """
        return self._comparison(other, np.ndarray.__gt__)

    def __le__(self, other):
        """
        Checks if Var object is less than or equal to another object.
        If other object is Var object, returns result of numpy comparison of their values.
        """
        return self._comparison(other, np.ndarray.__le__)

    def __ge__(self, other):
        """
        Checks if Var object is greater than or equal to another object.
        If other object is Var object, returns result of numpy comparison of their values.
        """
        return self._comparison(other, np.ndarray.__ge__)
