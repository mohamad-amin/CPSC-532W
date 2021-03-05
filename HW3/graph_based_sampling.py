import torch
import numpy as np
import pickle as pkl
from daphne import daphne
from p_tests import is_tol, run_prob_test, load_truth
from evaluation_based_sampling import evaluate_program
from copy import deepcopy
from torch.distributions.multivariate_normal import MultivariateNormal


def deterministic_eval_grad(ast):
    result = deterministic_eval(ast)
    if torch.is_tensor(result):
        return result
    else:
        return torch.tensor([float(result)], requires_grad=True).squeeze()


def deterministic_eval(ast):
    return evaluate_program([ast])


def topological_sort(nodes, edges):
    visited = {k: v for k, v in zip(nodes, [False] * len(nodes))}
    dfs = []
    result = []
    for node in nodes:
        if not visited[node]:
            dfs.append((False, node))
        while len(dfs) > 0:
            v, node = dfs.pop()
            if v:
                result.append(node)
                continue
            visited[node] = True
            dfs.append((True, node))
            if node not in edges:
                continue
            children = edges[node]
            for child in children:
                if not visited[child]:
                    dfs.append((False, child))
    return result[::-1]


def fill_vals(e, trace):
    if type(e) == list:
        return [fill_vals(exp, trace) for exp in e]
    else:
        return trace[e] if e in trace else e


def sample_from_joint_vars(graph):
    results = {}
    P = graph['P']
    tp_nodes = topological_sort(graph['V'], graph['A'])

    for node in tp_nodes:
        lnk = P[node][0]
        if lnk == "sample*":
            le = fill_vals(P[node][1], results)
            res = deterministic_eval(le).sample()
        elif lnk == "observe*":
            res = graph['Y'][node]
        else:
            pass
            # TODO: what?
        results[node] = res

    return results


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."

    results = {}
    P = graph[1]['P']
    tp_nodes = topological_sort(graph[1]['V'], graph[1]['A'])

    for node in tp_nodes:
        lnk = P[node][0]
        if lnk == "sample*":
            le = fill_vals(P[node][1], results)
            res = deterministic_eval(le).sample()
        elif lnk == "observe*":
            res = graph[1]['Y'][node]
        else:
            pass
            # TODO: what?
        results[node] = res

    return deterministic_eval(fill_vals(graph[2], results))


def calculate_X_Y_vars(graph):
    nodes = graph['V']
    X = []
    Y = []
    for v in nodes:
        if graph['P'][v][0] == "sample*":
            X.append(v)
        elif graph['P'][v][0] == "observe*":
            Y.append(v)
    return X, Y


def calculate_free_var_information(graph):
    flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]
    nodes = graph['V']
    free_vars = {}
    for v in nodes:
        link_exp = flatten(graph['P'][v][1])
        free_vars[v] = [v2 for v2 in link_exp if v2 in nodes and v2 != v]
    in_free_vars = {}
    for v in nodes:
        in_free_vars[v] = [v2 for v2 in free_vars if v in free_vars[v2]]
    return free_vars, in_free_vars


def mh_gibbs_accept(p, x, current_values, proposed_values, in_free_vars_dict, Y):
    d1 = deterministic_eval(fill_vals(p[x][1], current_values))
    d2 = deterministic_eval(fill_vals(p[x][1], proposed_values))
    logA = d2.log_probability(current_values[x]) - d1.log_probability(proposed_values[x])
    vx = in_free_vars_dict[x]
    for v in vx:
        logA += deterministic_eval(fill_vals(p[v][1], proposed_values)).log_probability(proposed_values[v])
        logA -= deterministic_eval(fill_vals(p[v][1], current_values)).log_probability(current_values[v])
    return np.exp(logA)


def mh_gibbs_step(p, X, values, in_free_vars_dict, Y):
    for x in X:
        d = deterministic_eval(fill_vals(p[x][1], values))
        proposed_values = values.copy()
        proposed_values[x] = d.sample()
        alpha = mh_gibbs_accept(p, x, values, proposed_values, in_free_vars_dict, Y)
        u = np.random.uniform()
        if u < alpha:
            values = proposed_values
    return values


def mh_gibbs(graph, xy_vars, free_var_information, steps):
    free_vars, in_free_vars = free_var_information
    X, Y = xy_vars
    values = [sample_from_joint_vars(graph)]

    for s in range(steps):
        values.append(mh_gibbs_step(graph['P'], X, values[-1], in_free_vars, Y))

    return values


def mh_gibbs_sampling(graph, steps=10000):
    fvi = calculate_free_var_information(graph[1])
    xy = calculate_X_Y_vars(graph[1])
    samples = []
    values = mh_gibbs(graph[1], xy, fvi, steps)
    for v in values:
        samples.append(deterministic_eval(fill_vals(graph[2], v)))
    return samples


def add_to_dict(d, a):
    v = {}
    for i, k in enumerate(list(d.keys())):
        v[k] = d[k].detach() + a[i]
        v[k].requires_grad = True
    return v


def hmc_u(X, Y, P):
    logP = 0
    for node in Y:
        logP += deterministic_eval(fill_vals(P[node][1], {**X, **Y})).log_probability(Y[node])
    return -logP


def hmc_h(X, r, M_inv, Y, P):
    return hmc_u(X, Y, P) + .5 * torch.matmul(r.T, torch.matmul(M_inv, r))


def hmc_grad(X, Y, P):
    Ux = hmc_u(X, Y, P)
    Ux.backward()
    gradients = torch.zeros(len(X))
    for i, key in enumerate(list(X.keys())):
        gradients[i] = X[key].grad
    return gradients


def hmc_leapfrog(X0, r0, T, eps, Y, P):
    r = r0 - .5 * eps * hmc_grad(X0, Y, P)
    X = X0
    for t in range(1, T):
        X = add_to_dict(X, eps * r)
        r = r - eps * hmc_grad(X, Y, P)
    X = add_to_dict(X, eps * r)
    r = r - .5 * eps * hmc_grad(X, Y, P)
    return X, r


def hamilton_monte_carlo(X, S, T, eps, M, Y, P):
    M_inv = M.inverse()
    r_dist = MultivariateNormal(torch.zeros(len(M)), M)
    samples = []

    for s in range(S):
        r = r_dist.sample()
        X_new, r_new = hmc_leapfrog(deepcopy(X), r, T, eps, Y, P)
        u = np.random.uniform()
        if u < torch.exp(-hmc_h(X_new, r_new, M_inv, Y, P) + hmc_h(X, r, M_inv, Y, P)):
            X = X_new
        samples.append(X)

    return samples


def hmc_sampling(graph, samples=int(1e4)):
    X, Y = calculate_X_Y_vars(graph[1])
    initial_values = sample_from_joint_vars(graph[1])
    Y = {k: initial_values[k] for k in initial_values if k in Y}
    X = {k: initial_values[k] for k in initial_values if k in X}
    for k in X:
        X[k] = (X[k] if torch.is_tensor(X[k]) else torch.tensor(X[k])).type(torch.float64)
        X[k].requires_grad = True

    T = 10
    eps = 0.1
    M = torch.eye(len(X))

    result = []
    samples = hamilton_monte_carlo(X, samples, T, eps, M, Y, graph[1]['P'])
    for X in samples:
        result.append(deterministic_eval(fill_vals(graph[2], X)))

    return result


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)


# Testing:
def extract_graphs():
    for i in range(1, 14):
        graph = daphne(['graph', '-i', '../HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        with open('programs/tests/deterministic/test_{}_graph.pkl'.format(i), 'wb') as f:
            pkl.dump(graph, f)
        print('Det {} done.'.format(i))
    for i in range(1, 7):
        graph = daphne(['graph', '-i', '../HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        with open('programs/tests/probabilistic/test_{}_graph.pkl'.format(i), 'wb') as f:
            pkl.dump(graph, f)
        print('Prob {} done.'.format(i))
    for i in range(1, 6):
        graph = daphne(['graph', '-i', '../HW2/programs/tests/hw3/program_{}.daphne'.format(i)])
        with open('programs/tests/hw3/program_{}_graph.pkl'.format(i), 'wb') as f:
            pkl.dump(graph, f)
        print('HW3 {} done.'.format(i))


def extract_model_graphs():
    for i in range(1, 5):
        graph = daphne(['graph', '-i', '../HW2/programs/{}.daphne'.format(i)])
        with open('programs/{}_graph.pkl'.format(i), 'wb') as f:
            pkl.dump(graph, f)


def run_deterministic_tests():
    for i in range(1, 13):
        # note: this path should be with respect to the daphne path!
        graph = pkl.load(open('programs/tests/deterministic/test_{}_graph.pkl'.format(i), 'rb'))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert (is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret, truth, graph))

        print('Test {} passed'.format(i))

    print('All deterministic tests passed')


def run_probabilistic_tests():
    num_samples = 1e4
    max_p_value = 1e-4

    for i in range(1, 7):
        # note: this path should be with respect to the daphne path!
        graph = pkl.load(open('programs/tests/probabilistic/test_{}_graph.pkl'.format(i), 'rb'))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(graph)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert (p_val > max_p_value)
        print('Test {} done'.format(i))

    print('All probabilistic tests passed')


def load_graph_env(i):
    return pkl.load(open('programs/tests/hw3/program_{}_graph.pkl'.format(i), 'rb'))


if __name__ == '__main__':

    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1, 5):
        graph = pkl.load(open('programs/{}_graph.pkl'.format(i), 'rb'))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))

# import sys; import os; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['/Users/amin/Projects/Masters/Winter2021/CPSC-532W/HW2'])
# os.chdir('HW2')
# from HW2.graph_based_sampling import *
