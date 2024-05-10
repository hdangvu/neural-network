# utils

import numpy as np
import matplotlib.pyplot as plt
from mat import *

# Abstract class
class MyDataset():
    '''
     MyDataset

     This is an abstract base class for other classes that store
     datasets. Derive another class from MyDataset, and override
     the functions __init__, __len__, and __getitem__.
    '''
    def __init__(self):
        return

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError



# DiscreteMapping: creates a simple classification dataset,
# mapping the row-vectors in A to the row-vectors in B.

class DiscreteMapping(MyDataset):
    '''
     DiscreteMapping(A, n=300, noise=0.1, random=True)

     Generates a DiscreteMapping dataset object with n samples.
     Each row in A corresponds to the mean of a class. The data sanoke
     is generated by selecting a row from A and adding noise to it.

     Inputs:
       A      input prototypes
       n      number of samples to generate
       noise  how much noise to add to the inputs
       random randomize the order of the samples

     Usage:
       ds = DiscreteMapping(A)
       ds.inputs()   # returns an array of input samples (one per row)
       ds.targets()  # returns an array of corresponding one-hot targets
       print(len(ds))  # the number of samples
       ds[5]    # returns the 5th sample, a list containing an input and its target
    '''
    def __init__(self, A, n=300, noise=0.1, random=True):
        self.n = n
        self.n_classes, self.input_dim = A.shape
        if random:
            self.T = np.random.randint(self.n_classes, size=(n,1))
        else:
            self.T  = (np.arange(n)%self.n_classes)[:,np.newaxis]

        self.X = A[self.T.squeeze()] + noise*np.random.randn(*A[self.T.squeeze()].shape)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]

    def __len__(self):
        return self.n

    def inputs(self):
        '''
         x = ds.inputs()

         Returns all the inputs of the dataset, with one sample
         on each row of the 2D array x.
        '''
        return self.X

    def targets(self):
        '''
         t = ds.targets()

         Returns all the targets of the dataset, with one sample
         on each row of the 2D array t.
        '''
        targets = np.zeros((self.n, self.n_classes))
        targets[np.arange(self.n), self.T[:,0]] = 1
        return targets

    def inputs_of_class(self, c):
        '''
         x = ds.inputs_of_class(c)

         Returns all the inputs of class c.

         Inputs:
           c   the index for the desired class

         Output:
           x   matrix, with one input sample on each row
        '''
        return self.X[np.argmax(self.T) == c]

    def class_means(self):
        '''
         ds.class_means()

         Returns the centroid of each class.
        '''
        return np.array([self.inputs_of_class(c).mean() for c in range(self.n_classes)])

    def plot(self, labels=None, idx=(0,1), equal=True):
        '''
         ds.plot(labels=None, idx=(0,1), equal=True)

         Plots the dataset as a scatterplot.

         Inputs:
          labels  a matrix containing a class vector in each row
                  Points are coloured according to the index of the
                  max element.
                  By default, the points are coloured according to the
                  dataset targets.
                  If labels has only 1 column, then it is interpretted
                  as the class index.
          idx     plots coord idx[0] on the horiz axis, and idx[1] on the
                  vertical axis.
          equal   Boolean: whether to use the same scale on the axes.
        '''
        colour_options = ['y', 'r', 'g', 'b', 'k']
        X = self.X
        x_index, y_index = idx
        if labels is None:
            cidx = self.T.squeeze()
        else:
            if len(labels[0])>1:
                # supplied labels are assumed to be one-hot vectors
                cidx = np.argmax(labels, axis=1)
            else:
                # supplied labels are class index
                cidx = np.clip((labels.squeeze()+0.5).astype(int), 0, self.n_classes-1)
        colours = [colour_options[k] for k in cidx]
        plt.scatter(X[:,idx[0]], X[:,idx[1]], color=colours, marker='.')
        if equal:
            plt.axis('equal')



class Composite(DiscreteMapping):
    '''
     ds = Composite(fcns, n=300)

     Creates a dataset object for classification. The data points are
     generated by the sampling functions in fcn.

     Inputs:
       fcns   list of functions that generate samples, one function per
              class. The functions must take no arguments, and return
              one sample as a list or array.
       n      number of samples to generate
       noise  how much noise to add to the dataset

     Usage:
       ds = Composite([fcn1, fcn2], n=100)
       ds.inputs()   # inputs (one sample per row)
       ds.targets()  # corresponding one-hot class vectors
       ds[4]  # returns 4th sample, a list with a numpy array, and a class index
    '''
    def __init__(self, fcns, n=300, noise=0.1, onehot=False):
        self.n = n
        x = fcns[0]()  # get a sample to know its dims
        self.n_classes = len(fcns)
        self.onehot = onehot
        self.input_dim = len(x)
        indexes = np.random.randint(self.n_classes, size=(n,1))
        self.X = np.array([fcns[i]() for i in indexes[:,0]])
        self.T = indexes



class Annuli(Composite):
    def __init__(self, n=300, noise=1.):
        self.r1 = 0.0
        self.r2 = 0.5
        self.r3 = 0.8

        def annulus(r, noise=0.1):
            theta = np.random.random()*2.*np.pi
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            return np.array([x,y]) + np.random.normal((0,0), (noise, noise))

        a1 = (lambda r=self.r1,n=0.1*noise: annulus(r,noise=n))
        a2 = (lambda r=self.r2,n=0.08*noise: annulus(r,noise=n))
        a3 = (lambda r=self.r3,n=0.1*noise: annulus(r,noise=n))
        super().__init__([a1, a2, a3], n=n)


class UClasses(Composite):

    def __init__(self, n=300):
        self.h = 1.
        self.w = 1.
        self.theta = np.random.random()*2.*np.pi
        self.offset = np.random.random(size=(2,))*4.-2#np.array([0.,0])
        M = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta), np.cos(self.theta)]])

        def u_shape(noise=0.1):
            '''
             p = u_shape(noise=0.1)
             Make a generic U-shaped cluster.
            '''
            arclength = 2*self.h + np.pi*self.w/2.
            r = np.random.random()*arclength
            if r<self.h:
                p = np.array([-self.w/2., self.h-r])
            elif r<self.h+np.pi*self.w/2.:
                theta = (r-self.h)/(self.w/2.) + np.pi
                p = self.w/2.*np.array([np.cos(theta), np.sin(theta)])
            else:
                p = np.array([self.w/2., self.h-(arclength-r)])
            return p + np.random.normal(scale=noise, size=(2,))

        def u_sample1(noise=0.1):
            '''
             p = u_sample1(noise=0.1)
             Make a rotated and shifted U cluster.
            '''
            p = u_shape(noise=noise)
            return p@M + self.offset

        def u_sample2(noise=0.1):
            '''
             p = u_sample1(noise=0.1)
             Make a flipped, counter-rotated and shifted U cluster.
            '''
            p = u_shape(noise=noise)
            return (p@np.array([[1,0],[0,-1]]) + np.array([self.w/2., self.h]))@M + self.offset

        super().__init__([u_sample1, u_sample2], n=n)






class ContinuousMapping(MyDataset):
    '''
     ContinuousMapping(fcn, domain, n=300, noise=0.1, random=True)

     Generates a ContinuousMapping dataset object with n samples, which
     can be used as a regression dataset.
     To create the dataset, a random point is chosen uniformly from the
     domain and fed through the function fcn. The targets are those
     outputs with noise added.

     Inputs:
       fcn     function, returning either a scalar or array-like
       domain  list of tuples, defining the domain for each of the
               input variables
               eg. if the input is 3D, each between -1 and 1, then
                   domain = [(-1,1),(-1,1),(-1,1)]
       n       number of samples to generate
       noise   how much noise to add to the outputs

     Usage:
       f = (lambda x: 1.+x[0]-0.5x[1]**2)
       ds = ContinuousMapping(f, [(-1,1), (-2,2)])
       ds.inputs()   # returns an array of input samples (one per row)
       ds.targets()  # returns an array of corresponding targets
       print(len(ds))  # the number of samples
       ds[5]    # returns the 5th sample, a list containing an input and its target
    '''
    def __init__(self, fcn, domain, n=300, noise=0.1):
        self.n = n
        self.domain = np.array(domain)
        self.domain_min = self.domain[:, 0]
        self.domain_max = self.domain[:, 1]
        self.input_dim = self.domain.shape[0]

        self.X = self.domain_min + (self.domain_max - self.domain_min)*np.random.rand(self.n, self.input_dim)
        self.X = np.float32(self.X)
        # Targets (this is supposed to work for 1-D or N-D)
        self.T = np.array([(fcn(x)) for x in self.X], dtype=np.float32)
        self.T += noise*np.random.randn(*self.T.shape)
        self.T = self.T.reshape(n, np.prod(self.T.shape)//n)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]

    def __len__(self):
        return self.n

    def inputs(self):
        '''
         x = ds.inputs()

         Returns all the inputs of the dataset, with one sample
         on each row of the 2D array x.
        '''
        return self.X

    def targets(self):
        '''
         t = ds.targets()

         Returns all the targets of the dataset, with one sample
         on each row of the 2D array t.
        '''
        return self.T

    def plot(self, x_idx=0, t_idx=0, equal=False):
        '''
         ds.plot(labels=[], idx=(0,1), equal=True)

         Plots the dataset as a scatterplot.

         Inputs:
          labels  a matrix containing a class vector in each row
                  Points are coloured according to the index of the
                  max element.
                  By default, the points are coloured according to the
                  dataset targets.
          idx     plots coord idx[0] on the horiz axis, and idx[1] on the
                  vertical axis.
          equal   Boolean: whether to use the same scale on the axes.
        '''
        X = self.inputs()
        T = self.targets()
        plt.scatter(X[:,x_idx], T[:,t_idx], marker='.')

        if equal:
            plt.axis('equal')




# MyDataLoader
class MyDataLoader():
    def __init__(self, ds, batchsize=1, shuffle=False):
        '''
         dl = MyDataLoader(ds, batchsize=1, shuffle=False)

         Creates an iterable dataloader object that can be used to feed batches into a neural network.

         Inputs:
           ds         a MyDataset object
           batchsize  size of the batches
           shuffle    randomize the ordering

         Then,
           next(dl) returns the next batch
                    Each batch is a tuple containing (inputs, targets), where:
                     - inputs is a 2D numpy array containing one input per row, and
                     - targets is a 2D numpy array with a target on each row
        '''
        self.index = 0
        self.ds = ds
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.n_batches = (len(ds)-1)//self.batchsize + 1   # might include a non-full last batch

        self.make_batches()

    def reset(self):
        '''
         MyDataLoader.reset()

         Resets the pointer to the beginning of the dataset.
        '''
        self.index = 0
        if self.shuffle:
            self.make_batches()

    def make_batches(self):
        self.order = np.arange(len(self.ds))
        if self.shuffle:
            np.random.shuffle(self.order)

        X, Y = self.ds[self.order]

        split_indices = np.arange(self.batchsize, len(self.ds), step=self.batchsize) # where to cut into batches

        self.X_batches = np.split(X, split_indices)
        self.Y_batches = np.split(Y, split_indices)

    def __next__(self):
        '''
         Outputs:
           dl       the next batch
        '''
        if self.index < self.n_batches:
            X = self.X_batches[self.index]
            Y = self.Y_batches[self.index]
            self.index += 1
            return X, Y
        raise StopIteration

    def __iter__(self):
        return self


class MatUClasses(UClasses):
    '''
     mds = MatUClasses(n=1000)
     
     Creates a MatUClasses dataset that produces Mat objects.
     
     Usage:
      mds = MatUClasses(n=100)
      mds.plot()
      mds.inputs()  # -> Mat object
      mds.targets() # -> Mat object
    '''
    def inputs(self):
        return Mat(super().inputs())
    def targets(self):
        return Mat(super().targets())
    def plot(self, labels=None, **kwargs):
        if labels is None:
            super().plot(**kwargs)
        else:
            super().plot(labels=labels.val, **kwargs)
            
class MatAnnuli(Annuli):
    '''
     mds = MatAnnuli(n=1000)
     
     Creates a MatAnnuli dataset that produces Mat objects.
     
     Usage:
      mds = MatAnnuli(n=100)
      mds.plot()
      mds.inputs()  # -> Mat object
      mds.targets() # -> Mat object
    '''
    def inputs(self):
        return Mat(super().inputs())
    def targets(self):
        return Mat(super().targets())
    def plot(self, labels=None, **kwargs):
        if labels is None:
            super().plot(**kwargs)
        else:
            super().plot(labels=labels.val, **kwargs)




#end