import torch
import pickle as pkl
from daphne import daphne
from p_tests import is_tol, run_prob_test, load_truth
from primitives import funcprimitives, distributions, Function


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    defs = {}
    if isinstance(ast[0], list) and ast[0][0] == "defn":
        while ast[0][0] == "defn":
            defs[ast[0][1]] = Function(ast[0][1], ast[0][2], ast[0][3])
            ast = ast[1]
    else:
        ast = ast[0]
    res = eval(ast, {}, {}, defs)
    return res[0]


def eval(e, sigma, L, defs):
    if type(e) != list or len(e) == 1:
        if type(e) == list:
            e = e[0]
        if type(e) == int or type(e) == float:
            return torch.tensor(float(e)), sigma
        elif e in funcprimitives:
            return e, sigma
        elif e in distributions:
            return e, sigma
        elif e in defs:
            return e, sigma
        elif torch.is_tensor(e):
            return e, sigma
        else:
            # variable
            return L[e], sigma
    elif e[0] == "sample":
        dist, sigma = eval(e[1], sigma, L, defs)
        return dist.sample(), sigma
    elif e[0] == "observe":
        return None, sigma
    elif e[0] == "let":
        exp, sigma = eval(e[1][1], sigma, L, defs)
        for e in e[2:-1]:
            _, sigma = eval(e, sigma, {**L, **{e[1][0]: exp}}, defs)
        return eval(e[-1], sigma, {**L, **{e[1][0]: exp}}, defs)
    elif e[0] == "if":
        cond, sigma = eval(e[1], sigma, L, defs)
        if cond:
            exp, sigma = eval(e[2], sigma, L, defs)
        else:
            exp, sigma = eval(e[3], sigma, L, defs)
        return exp, sigma
    else:
        cs = []
        for exp in e:
            c, sigma = eval(exp, sigma, L, defs)
            cs.append(c)
        if cs[0] in funcprimitives:
            return funcprimitives[cs[0]](cs[1:]), sigma
        elif cs[0] in distributions:
            return distributions[cs[0]](cs[1:]), sigma
        elif cs[0] in defs:
            func = defs[cs[0]]
            args = {k: v for k, v in zip(func.variables, cs[1:])}
            return eval(func.proc, sigma, {**L, **args}, defs)
        else:
            pass
            # TODO: what?


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


def extract_asts():
    for i in range(1, 14):
        ast = daphne(['desugar', '-i', '../HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        with open('programs/tests/deterministic/test_{}_ast.pkl'.format(i), 'wb') as f:
            pkl.dump(ast, f)
        print('Det {} done.'.format(i))
    for i in range(1, 7):
        ast = daphne(['desugar', '-i', '../HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        with open('programs/tests/probabilistic/test_{}_ast.pkl'.format(i), 'wb') as f:
            pkl.dump(ast, f)
        print('Prob {} done.'.format(i))


def extract_model_asts():
    for i in range(1, 5):
        ast = daphne(['desugar', '-i', '../HW2/programs/{}.daphne'.format(i)])
        with open('programs/{}_ast.pkl'.format(i), 'wb') as f:
            pkl.dump(ast, f)


def run_deterministic_tests():
    for i in range(1, 14):
        # note: this path should be with respect to the daphne path!
        ast = pkl.load(open('programs/tests/deterministic/test_{}_ast.pkl'.format(i), 'rb'))
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast), None
        try:
            assert (is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast))

        print('Test {} passed'.format(i))

    print('All deterministic tests passed')


def run_probabilistic_tests():
    num_samples = 1e4
    max_p_value = 1e-4

    for i in range(1, 7):
        # note: this path should be with respect to the daphne path!
        ast = pkl.load(open('programs/tests/probabilistic/test_{}_ast.pkl'.format(i), 'rb'))
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert (p_val > max_p_value)
        print('Test {} done'.format(i))

    print('All probabilistic tests passed')


if __name__ == '__main__':

    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1, 5):
        ast = pkl.load(open('programs/{}_ast.pkl'.format(i), 'rb'))
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))
