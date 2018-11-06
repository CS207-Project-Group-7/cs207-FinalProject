import math
import numpy as np
import collections
import numbers

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
        result = len(args) * [0]
        for i, var in enumerate(args):
            self._compute_grad(var)
            result[i] = self._grad_cache[var]
        return np.array(result)
    
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
    
    def __add__(self, other):
        """
        Returns Scalar object representing addition of two Scalar objects or
        the addition of a Scalar object and a Python number.
        """
        try:
            result = Scalar(self.val + other.val)
            result.parents.append((1., other))
        except AttributeError:
            result = Scalar(self.val + other)
        result.parents.append((1., self))
        return result

    def __radd__(self, other):
        """
        Returns Scalar object representing right addition of a Scalar object
        with a Python number.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Returns Scalar object representing subtraction of two Scalar objects or
        the subtraction of a Scalar object and a Python number
        """
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        """
        Returns Scalar object representing right subtraction of a Scalar object
        with a Python number
        """
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        """
        Returns Scalar object representing multiplication of two Scalar objects 
        or the multiplication of a Scalar object and a Python number
        """
        try:
            result = Scalar(self.val * other.val)
            result.parents.append((other.val, self))
            result.parents.append((self.val, other))
        except AttributeError:
            result = Scalar(other * self.val)
            result.parents.append((other, self))
        except TypeError:
            result = other * self
        return result
    
    def __rmul__(self, other):
        """
        Returns Scalar object representing right multiplication of a Scalar 
        object with a Python number
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Returns Scalar object representing division of two Scalar objects or 
        the division of a Scalar object and a Python number
        """
        return self.__mul__(other.__pow__(-1))

    def __rtruediv__(self, other):
        """
        Returns Scalar object representing right division of a Scalar object
        with a Python number
        """
        return self.__pow__(-1).__mul__(other)

    def __pow__(self, other):
        """
        Returns Scalar object representing exponentiation of two Scalar objects 
        or the exponentiation of a Scalar object and a Python number
        """
        try:
            result = Scalar(self.val ** other.val)
            result.parents.append((other.val * self.val ** (other.val - 1), self))
            result.parents.append((math.log(self.val) * self.val ** other.val, other))
        except AttributeError:
            result = Scalar(math.pow(self.val, other))
            result.parents.append((other * self.val ** (other - 1), self))
        return result

    def __rpow__(self, other):
        """
        Returns Scalar object representing right exponentiation of a Scalar 
        object with a Python number
        """
        result = Scalar(other ** self.val)
        result.parents.append((math.log(other) * other ** self.val, self))
        return result

    def __abs__(self):
        """
        Returns Scalar object representing absolute value of a Scalar object
        """
        result = Scalar(abs(self.val))
        result.parents.append((self.val / abs(self.val), self))
        return result

    def _ban_augmented_assignment(self):
        """
        Internal method for raising error when in-place operation with Scalar
        objects is attempted.
        """
        raise TypeError("In-place operations are not supported for lazydiff variables.")
    
    def __iadd__(self, other):
        """
        Raises error when in-place addition is called
        """
        self._ban_augmented_assignment()

    def __isub__(self, other):
        """
        Raises error when in-place subtraction is called
        """
        self._ban_augmented_assignment()

    def __imul__(self, other):
        """
        Raises error when in-place multiplication is called
        """
        self._ban_augmented_assignment()

    def __itruediv__(self, other):
        """
        Raises error when in-place division is called
        """
        self._ban_augmented_assignment()

    def __ipow__(self, other):
        """
        Raises error when in-place exponentiation is called
        """
        self._ban_augmented_assignment()


class Vector:
    def __init__(self, *args):
        if isinstance(args[0],Scalar):
            self._components = args
            self.val = tuple([component.val for component in self._components])
        # support list or ndarray input
        elif ((isinstance(args[0],collections.Sequence) or isinstance(args[0],np.ndarray)) and len(args)==1):
            self._components = tuple(args[0])
            try:
                self.val = tuple([component.val for component in self._components])
            except AttributeError:
                raise TypeError('sequence elements must be of type Scalar')
        else:
            raise TypeError("inputs need to be Scalar objects or a sequence of Scalar objects")

    def grad(self, *args):
        if (args == ()): 
            raise ValueError('Must pass value(s) to take respct with')
        if (isinstance(args[0],Scalar)):
            return np.array([component.grad(*args) for component in self._components])
        # elementwise or all components?
        elif (isinstance(args[0],Vector) and len(args)==1):
            # elementwise
            # return tuple([comp1.grad(comp2) for comp1, comp2 in zip(self._components, args[0]._components)])
            # Hessian
            return np.array([comp1.grad(*args[0]) for comp1 in self._components])

    def __getitem__(self, ind):
        if ind not in range(self.__len__()):
            raise IndexError
        return self._components[ind]

    def __len__(self):
        return len(self._components)

    def _check_broadcast(self, other):
        if (len(self) != len(other)):
            raise ValueError("operands could not be broadcast together with lengths {} and {}".format(len(self),len(other)))

    def __neg__(self):
        return Vector([-component for component in self._components])
    
    def __add__(self, other):
        if (isinstance(other, numbers.Number) or isinstance(other, Scalar)):
            return Vector([component + other for component in self._components])
        elif (isinstance(other, Vector)):
            self._check_broadcast(other)
            return Vector([comp1+comp2 for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("input needs to be a numeric value or Vector object")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if (isinstance(other, numbers.Number) or isinstance(other, Scalar)):
            return Vector([component*other for component in self._components])
        elif (isinstance(other, Vector)):
            self._check_broadcast(other)
            return Vector([comp1*comp2 for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("input needs to be Scalar or Vector object")
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**(-1)

    def __rtruediv__(self, other):
        return self**(-1) * other

    def __pow__(self, other):
        return Vector([component**other for component in self._components])
