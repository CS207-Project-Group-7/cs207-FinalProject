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
        if args == (): 
            raise ValueError('Must pass value(s) to take gradient to respect with')
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
            result = Scalar(math.pow(self.val, other))
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
        if isinstance(args[0], Scalar):
            self._components = args
            self.val = tuple([component.val for component in self._components])
        # support list or ndarray input
        elif ((isinstance(args[0], collections.Sequence) or isinstance(args[0], np.ndarray)) and len(args) == 1):
            self._components = tuple(args[0])
            try:
                self.val = tuple([component.val for component in self._components])
            except AttributeError:
                raise TypeError('Sequence elements must be of type Scalar')
        else:
            raise TypeError("inputs need to be Scalar objects or a sequence of Scalar objects")

    def grad(self, *args):
        if (args == ()): 
            raise ValueError('Must pass value(s) to take gradient to respect with')
        if (isinstance(args[0], Scalar)):
            return np.array([component.grad(*args) for component in self._components])
        elif (isinstance(args[0], Vector) and len(args) == 1):
            return np.array([comp1.grad(*args[0]) for comp1 in self._components])

    def __getitem__(self, ind):
        if ind not in range(len(self)):
            raise IndexError
        return self._components[ind]

    def __len__(self):
        return len(self._components)

    def _check_broadcast(self, other):
        if (len(self) != len(other)):
            raise ValueError("Operands could not be broadcast together with lengths {} and {}".format(len(self),len(other)))

    def __neg__(self):
        return Vector([-component for component in self._components])
    
    def __add__(self, other):
        if isinstance(other, numbers.Number) or isinstance(other, Scalar):
            return Vector([component + other for component in self._components])
        elif isinstance(other, Vector):
            self._check_broadcast(other)
            return Vector([comp1 + comp2 for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("Input needs to be a numeric value or Vector object")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, numbers.Number) or isinstance(other, Scalar):
            return Vector([component * other for component in self._components])
        elif isinstance(other, Vector):
            self._check_broadcast(other)
            return Vector([comp1 * comp2 for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("Input needs to be Scalar or Vector object")
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __pow__(self, other):
        if isinstance(other, numbers.Number) or isinstance(other, Scalar):
            return Vector([component ** other for component in self._components])
        elif isinstance(other, Vector):
            self._check_broadcast(other)
            return Vector([comp1 ** comp2 for comp1, comp2 in zip(self._components, other._components)])
        else:
            raise TypeError("Input needs to be Scalar or Vector object")

    def __rpow__(self, other):
        return Vector([other ** component for component in self._components])