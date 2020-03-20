# Date: Wednesday 26 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: helper functions

import math

import torch

RAND_MAX = 2147483647


def sample_rand_uniform():
    """TODO: Docstring for sample_rand_uniform.

    :arg1: TODO
    :returns: TODO

    """
    # return ((random.randint(0, RAND_MAX) + 1) * 1.0) / (RAND_MAX + 2)
    rand_num = torch.randint(RAND_MAX, (1, 1)).item()
    return ((rand_num + 1) / (RAND_MAX + 2))
    # return torch.rand(1).item()


def sample_exp_two_sides(lambda_):
    """TODO: Docstring for sample_exp_two_sides.
    :returns: TODO

    """

    # pos_or_neg = random.randint(0, RAND_MAX)
    pos_or_neg = torch.randint(RAND_MAX, (1, 1)).item()
    if (pos_or_neg % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1

    rand_uniform = sample_rand_uniform()
    return math.log(rand_uniform) / (lambda_ * pos_or_neg)


if __name__ == "__main__":
    # out = sample_rand_uniform()
    # torch.manual_seed(800)
    # out = torch.rand(1)
    for i in range(1000000000000000000):
        print(sample_exp_two_sides(0.4))
