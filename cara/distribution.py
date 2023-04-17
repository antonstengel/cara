import numpy as np
import numpy.random as npr
import scipy

class Distribution(scipy.stats.rv_continuous):
    """Represents continuous distributions."""

    def __init__(self, dist: str, a: float, b: float):
        super().__init__(self)

        self.dist = dist
        # support ends
        self.a = a
        self.b = b

        self._validate_support()
        self._validate_cdf()

    def _cdf(self, x):
        return eval(self.dist) # super unsafe

    def _validate_support(self) -> bool:
        if self.a >= self.b:
            raise ValueError('Support is crossed.')
        if self.a < 0:
            raise ValueError('Support must be positive')
        return True

    def _validate_cdf(self) -> bool:
        # first checking correct set of symbols
        allowed = set('123456789+-/*()x')
        if set(self.dist) > allowed:
            raise ValueError('CDF contains bad characters.')

        # grabbing a valid point in the support
        if np.isinf(self.b):
            x = float(self.a + 1)
        else:
            x = (self.b + self.a) / 2

        # checking that input evaluates to a float
        try:
            assert(isinstance(eval(self.dist), float))
        except:
            raise ValueError('CDF does not evalaute to float.')

        # checking that the cdf is within proper bounds
        if eval(self.dist) > 1 or eval(self.dist) < 0:
            raise ValueError('CDF is outside of [0,1]')

    def discretize(self, type: str, support_size: int, parameters: list) -> tuple:
        """Takes a string name of a type of discretization. 
        Returns a tuple (masses, support) of lists.
        """
        if type == 'quantiles' or type == 'q':
            return self._quantiles_discretization(support_size, parameters)
        elif type == 'random' or type == 'r':
            return self._random_discretization(support_size, parameters)
        elif type == 'fixed' or type == 'f':
            return self._fixed_discretization(support_size, parameters)
        elif type == 'random-with-fixed' or type == 'rf':
            return self._random_with_fixed_discretization(support_size, parameters)
        else:
            raise ValueError('Invalid type of discretization.')

    def _quantiles_discretization(self, support_size: int, parameters: list) -> tuple:
        """support_size many evenly-spaced quantiles of the CDF. No parameters.
        """
        increment = 1 / support_size
        support = [self.a]
        support += [self.ppf(increment*i) for i in range(1, support_size)]
        support = sorted(support, reverse=True)

        masses = self.stochastically_dominated_masses(support)
        return masses, support

    # this is contains the functionality of _naive_random_discretization and _one_outlier_naive_random_discretization
    def _random_discretization(self, support_size: int, parameters: list) -> tuple:
        """Takes support_size-1 many points randomly between l1 and u1 as well as a point at self.a.
        
        Parameters:
        l1 u1
        """
        l1 = float(parameters[0])
        u1 = float(parameters[1])

        if l1 >= u1:
            raise ValueError('Support is crossed.')

        support = np.concatenate([npr.uniform(l1, u1, support_size-1), [self.a]])
        support = sorted(support, reverse=True)
        
        masses = self.stochastically_dominated_masses(support)
        return masses, support  

    def _fixed_discretization(self, support_size: int, parameters: list) -> tuple:
        """Takes a fixed support.

        Parameters:
        support[0] ... support[support_size-1]
        """
        if len(parameters) != support_size:
            raise ValueError('Wrong number of points in support.')
            
        support = np.array([float(x) for x in parameters])
        support = sorted(support, reverse=True)

        masses = self.stochastically_dominated_masses(support)
        return masses, support
    
    def _random_with_fixed_discretization(self, support_size: int, parameters: list) -> tuple:
        """Uses fixed points and a variable number of regions from which to randomly draw.
        
        Parameters:
        fixed_sup_size rand_sup_size  
        num_rand_groups rand_sizes[0] ... rand_sizes[num_rand_groups-1]  
        lower_sups[0] upper_sups[0] ... lower_sups[num_rand_groups-1] upper_sups[num_rand_groups-1]
        fixed_sup[0] ... fixed_sup[fixed_support_size-1]
        """

        fixed_sup_size = int(parameters[0])
        rand_sup_size = int(parameters[1])
        num_rand_groups = int(parameters[2])
        rand_sizes = [int(x) for x in parameters[3:3+num_rand_groups]]

        lower_sups, upper_sups = [], []
        idx = 3+num_rand_groups
        for i in range(0, 2*num_rand_groups, 2):
            lower_sups.append(float(parameters[idx+i+0]))
            upper_sups.append(float(parameters[idx+i+1]))

        idx += 2*num_rand_groups
        fixed_sup = parameters[idx:]

        # validating input
        if fixed_sup_size + rand_sup_size != support_size:
            raise ValueError('Support sizes do not add up.')
        if num_rand_groups != len(rand_sizes):
            raise ValueError('Incorrect number of random groups.')
        if rand_sup_size != sum(rand_sizes):
            raise ValueError('Incorrect number of random points.')
        if fixed_sup_size != len(fixed_sup):
            raise ValueError('Wrong number of fixed points in support.')
        for l, u in zip(lower_sups, upper_sups):
            if l >= u:
                raise ValueError('Support is crossed.')

        # putting support together
        support = []
        support += fixed_sup
        for l, u, n in zip(lower_sups, upper_sups, rand_sizes):
            support += npr.uniform(l, u, n).tolist()
        support = np.array(support, dtype=np.float64)
        support = sorted(support, reverse=True)

        masses = self.stochastically_dominated_masses(support)
        return masses, support

    def stochastically_dominated_masses(self, support: list):
        """ Takes a list of reverse-sorted values in the support, and returns the masses
        if you collapse all the probability leftwards to each support value. Results
        in the closest-fitting stochastically-dominated masses for a given discretized support.
        """
        if not all(support[i] >= support[i+1] for i in range(len(support) - 1)):
            raise ValueError('Support must be reverse sorted.')
            
        masses = []
        masses.append(1-self.cdf(support[0]))
        for i in range(1, len(support)):
            masses.append(self.cdf(support[i-1]) - self.cdf(support[i]))
        return masses
    
    def stochastically_dominating_masses(self, support: list):
        """ This function takes a descending list of values in the support, and returns the 
        masses if you expand all the probability rightwards to each support value. Results in 
        the closest-fitting stochastically-dominating masses for a given discretized support.
        """
        if not all(support[i] >= support[i+1] for i in range(len(support) - 1)):
            raise ValueError('Support must be reverse sorted.')
            
        masses = []
        # this should be 0 if it's properly stochastically-dominating
        # however, if the support is infinite, can only be approximately stochastically-dominating
        leftover = 1 - self.cdf(support[0])
        
        for i in range(1, len(support)): # by taking out the first append, everything shifted to right 1
            masses.append(self.cdf(support[i-1]) - self.cdf(support[i]))
        masses.append(self.cdf(support[-1]))
        masses[0] += leftover
        return masses