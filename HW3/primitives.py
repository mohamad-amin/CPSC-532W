import torch
from torch import distributions as dists


def put(a):
    if type(a[0]) != dict:
        a[0][a[1].long()] = a[2]
        return a[0]
    else:
        a[0][a[1].item()] = a[2]
        return a[0]


def get(a):
    if type(a[0]) != dict:
        return a[0][a[1].long()]
    else:
        return a[0][a[1].item()]


def make_map(a):
    keys, vals = a[::2], a[1::2]
    keys = list(map(lambda x: x.item(), keys))
    return dict(zip(keys, vals))


def make_vector(a):
    try:
        return torch.stack(a)
    except:
        return a


funcprimitives = {
    '/': lambda a: a[0] / a[1],
    '*': lambda a: a[0] * a[1],
    '+': lambda a: a[0] + a[1],
    '-': lambda a: a[0] - a[1],
    '==': lambda a: a[0] == a[1],
    '=': lambda a: a[0] == a[1],
    '>': lambda a: a[0] > a[1],
    '<': lambda a: a[0] < a[1],
    '>=': lambda a: a[0] >= a[1],
    '<=': lambda a: a[0] <= a[1],
    'or': lambda a: a[0] or a[1],
    'and': lambda a: a[0] and a[1],
    'sqrt': lambda a: torch.sqrt(a[0]),
    'vector': make_vector,
    'get': get,
    'put': put,
    'first': lambda a: a[0][0],
    'second': lambda a: a[0][1],
    'last': lambda a: a[0][-1],
    'rest': lambda a: a[0][1:],
    'append': lambda a: torch.tensor(list(a[0]) + [a[1]]),
    'hash-map': make_map,
    'mat-transpose': lambda a: a[0].T,
    'mat-tanh': lambda a: a[0].tanh(),
    'mat-mul': lambda a: torch.matmul(a[0], a[1]),
    'mat-add': lambda a: a[0] + a[1],
    'mat-repmat': lambda a: a[0].repeat((int(a[1].item()), int(a[2].item())))
}


class Dist:

    def __init__(self, dist, name, params_count, *p):
        self.dist = dist
        self.name = name
        self.params_count = params_count
        self.params = []
        for i in range(params_count):
            self.params.append(p[i])

    def __str__(self):
        return str([self.name] + self.params)

    def sample(self):
        return self.dist.sample()

    def log_probability(self, c):
        return self.dist.log_prob(c)


class Normal(Dist):

    def __init__(self, a):
        dist = dists.Normal(a[0], a[1])
        super().__init__(dist, "normal", 2, a[0], a[1])


class Beta(Dist):

    def __init__(self, a):
        dist = dists.Beta(a[0], a[1])
        super().__init__(dist, "beta", 2, a[0], a[1])


class Exponential(Dist):

    def __init__(self, a):
        dist = dists.Exponential(a[0])
        super().__init__(dist, "exponential", 2, torch.tensor(0.), 1 / a[0])


class Uniform(Dist):

    def __init__(self, a):
        dist = dists.Uniform(a[0], a[1])
        super().__init__(dist, "uniform", 2, a[0], a[1])


class Discrete(Dist):

    def __init__(self, a):
        dist = dists.Categorical(a[0])
        super().__init__(dist, "discrete", 0)


class Gamma(Dist):

    def __init__(self, a):
        dist = dists.Gamma(a[0], a[1])
        super().__init__(dist, "Gamma", 2, a[0], a[1])


class Dirichlet(Dist):

    def __init__(self, a):
        dist = dists.Dirichlet(a[0])
        super().__init__(dist, "Dirichlet", -1, a)


class Bernoulli(Dist):

    def __init__(self, a):
        dist = dists.Bernoulli(a[0])
        super().__init__(dist, "Bernoulli", 1, a[0])


# class Dirac(Dist):
#
#     # def __init__(self, a):
#         # dist = dists.Dir(a[0])
#         # super().__init__(dist, "Bernoulli", 1, a[0])


distributions = {
    'normal': Normal,
    'beta': Beta,
    'exponential': Exponential,
    'uniform': Uniform,
    'discrete': Discrete,
    'gamma': Gamma,
    'dirichlet': Dirichlet,
    'flip': Bernoulli
}


class Function:

    def __init__(self, name, variables, proc):
        self.name = name
        self.variables = variables
        self.proc = proc
