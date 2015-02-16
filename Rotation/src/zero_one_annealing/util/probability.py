#!/usr/bin/python
"""probability utility classes and functions"""
import random

class ConditionalSampling:
    """A class to get i.i.d. samples from a conditional probability distribution p(y|x)."""
    def __init__(self, probability):
        """
        :param probability: a dictionary representing p(y | x), where
            key = x
            value = a dictionary in the form {y1: prob1, y2: value2, ...}
        """
        self.probability = probability

    def sample(self, x):
        r = random.uniform(0, 1)
        s = 0
        the_item = (0, 0, 0)
        for item in self.probability[x]:
            prob = self.probability[x][item]
            s += prob
            if s >= r:
                the_item = item
                break
        return the_item


def indicator(statement):
    """
    :param statement: a bool inside the indicator function
    :returns: 0 if false, 1 if true

    The function is really simple, but it is meant to make code more readable.
    """
    return int(statement)


def hamming(x, y):
    """x and y must have the same length, else it will cut off the tail of the longer one."""
    total = 0
    min_length = np.min([len(x), len(y)])
    for i in range(min_length):
        total += np.abs(x[i] - y[i])
    return total