# mat.py
# (C) Jeff Orchard, University of Waterloo, 2024

import numpy as np
import copy



'''
=========================================

 Mat class

=========================================
'''
class Mat(object):
    def __init__(self, val: np.ndarray):
        '''
         v = Mat(val)

         Creates a Mat object for the 2D numpy array val.

         Inputs:
           val   a 2D numpy array, dimensions DxN

         Output:
           v     object of Mat class

         Then, we can get its value using any of:
           v.val
           v()
         both of which return a numpy array.

         The member v.creator is either None, or is a reference
         to the MatOperation object that created v.

         The object also stores gradient information in v.grad.

         Usage:
           v = Mat(np.array([[1,2],[3,4.]]))
           len(v)  # returns number of rows
           v()     # returns v.val (a numpy array)
        '''
        self.val = np.array(copy.deepcopy(val), ndmin=2)
        self.rows, self.cols = np.shape(self.val)
        self.grad = np.zeros_like(self.val)
        self.creator = None

    def set(self, val: np.ndarray):
        self.val = np.array(val, dtype=float, ndmin=2)

    def zero_grad(self):
        self.grad = np.zeros_like(self.val)
        if self.creator!=None:
            self.creator.zero_grad()

    def backward(self, s: np.array =None):
        if s is None:
            s = np.ones_like(self.val)
        self.grad = self.grad + s
        if self.creator!=None:
            self.creator.backward(s)

    def __len__(self) -> int:
        return self.rows

    def __call__(self) -> np.ndarray:
        return self.val

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return f'Mat({self.val})'




'''
=========================================

 MatOperation

=========================================
'''
class MatOperation():
    '''
     op = MatOperation()

     MatOperation is an abstract base class for mathematical operations
     on matrices... Mat objects in particular.

     The MatOperation object op stores its arguments in the list op.args,
     and has the functions:
       op.__call__()
       op.zero_grad()
       op.backward()

     Usage:
       op()  # evaluates the operation without re-evaluating the args
       op.zero_grad() # resets grad to zero for all the args
       op.backward(s)  # propagates the derivative to the Vars below
    '''
    def __init__(self):
        self.args = []

    def __call__(self):
        raise NotImplementedError

    def zero_grad(self):
        for a in self.args:
            a.zero_grad()

    def backward(self, s=1.):
        raise NotImplementedError


'''
=========================================

 Operation Implementations

=========================================
'''

class Plus(MatOperation):
    '''
     Implements adding two Mat objects.
     Usage:
     > fcn = Plus()
     > C = fcn(A, B)  # performs C = A + B
    '''
    def __call__(self, a: Mat, b: Mat) -> Mat:
        self.args = [a, b]
        v = Mat(self.args[0].val + self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*np.ones_like(self.args[0].val))
        self.args[1].backward(s*np.ones_like(self.args[1].val))

class Minus(MatOperation):
    '''
     Implements subtracting two Mat objects.
     Usage:
     > fcn = Minus()
     > C = fcn(A, B)  # performs C = A - B
    '''
    def __call__(self, a: Mat, b: Mat) -> Mat:
        self.args = [a, b]
        v = Mat(self.args[0].val - self.args[1].val)
        v.creator = self
        return v

    def backward(self, s=None):
        self.args[0].backward(s*np.ones_like(self.args[0].val))
        self.args[1].backward(-s*np.ones_like(self.args[1].val))

class ReLU(MatOperation):
    '''
     Implements the ReLU activation function for Mat objects.
     Usage:
     > fcn = ReLU()
     > C = fcn(A)  # A and C are Mat objects
    '''
    def __call__(self, x: Mat) -> Mat:
        self.args = [x]
        v = Mat(np.clip(self.args[0].val, 0, None))
        v.creator = self
        return v

    def backward(self, s=1.):
        val = np.ceil( np.clip(self.args[0].val, 0, 1) )
        self.args[0].backward(s*val)

class Softmax(MatOperation):
    '''
     af = Softmax()

     Creates a MatOperation object that represents the softmax
     function. The softmax is applied to the rows of the input.

     Usage:
      > af = Softmax()
      > y = af( Mat([[0., 0.5]]) )
      > y.backward( np.array([1,0]) )
    '''
    def __call__(self, x: Mat) -> Mat:
        self.args = [x]
        num = np.exp(self.args[0].val) # Compute exp
        # Normalize by the sum
        denom = np.sum(num, axis=1)
        self.y = num / np.tile(denom[:,np.newaxis], [1,np.shape(num)[1]])
        v = Mat(self.y)
        v.creator = self
        return v

    def backward(self, s):
        '''
         af.backward(s)

         Computes and the derivative of the softmax function.
         Note that the __call__ function must be called before this
         function can be called.

         Input:
           s       NumPy array the same size as the input to __call__, which
                   multiplies the derivative
                   NOTE: s is a mandatory argument (not optional)
                   NOTE: s should have only a single non-zero element
                         in each row
        '''
        # This next line picks out the column index for all
        # the non-zero elements in s. Each row of s should have
        # only 1 non-zero element.
        idx = np.nonzero(s)[1]
        # deriv for one sample (row) is
        #   dE/dz_j = s_g * y_g * (delta_jg - y_j)
        # where g is the correct class for a given sample
        y = self.y
        s_gamma = np.zeros_like(s)
        y_gamma = np.zeros_like(y)
        kronecker = np.zeros_like(s)
        # Generate the required pieces
        for k,gamma in enumerate(idx):
            s_gamma[k,:] = s[k,gamma]
            y_gamma[k,:] = y[k,gamma]
            kronecker[k,gamma] = 1.
        dydz = s_gamma*y_gamma*(kronecker-y)  # dE/dz formula
        self.args[0].backward(dydz)
        #end not SOLUTIONS



#======== Loss Functions ========

class MSE(MatOperation):
    '''
     mserr = MSE()

     Creates a mean squared error function, with the following specifications:
      loss = mserr(y, target)

     where
      y       Mat object with output vectors in rows
      target  Mat object with one-hot target vectors in rows

     Returns the mean squared error as a (1,1) Mat object.

     Usage:
     > loss = MSE()(y, target)
     > loss.zero_grad()
     > loss.backward()

     Note that backward does not propagate the gradient for the target.
    '''
    def __call__(self, y: Mat, target: Mat) -> Mat:
        self.args = [y]
        self.target = target
        self.diff = self.args[0].val - self.target.val
        v = Mat( 0.5 * np.sum(self.diff**2, keepdims=True) / len(y) )
        v.creator = self
        return v

    def backward(self, s=1.):
        self.args[0].backward(s*self.diff/self.args[0].rows)








#end