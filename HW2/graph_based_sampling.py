import torch
import pickle as pkl
from daphne import daphne
from p_tests import is_tol, run_prob_test, load_truth
from evaluation_based_sampling import evaluate_program


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


if __name__ == '__main__':

    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1, 5):
        graph = pkl.load(open('programs/{}_graph.pkl'.format(i), 'rb'))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))
