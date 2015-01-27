#!/usr/bin/python
# usage: python toy.py theta0 theta1 theta2 theta3
import sys
import itertools
import math
import random
import simplejson
import numpy as np

k = 3
n = 10
m = 9

learning_verbose = False
prediction_verbose = False
true_theta = np.array([0.0, 0.0, 0.0, 0.0]) # to be initialized
X = range(1, n + 1)
Y = range(m + 1)
Z = {}
probability = {}
num_samples_per_x = 15
samples = []
samples_by_x = {}

step_size = 0.1

def main():
    if(len(sys.argv) != 5):
        print "must have exactly 4 arguments for the 4 values of theta"
        sys.exit(2)
    else:
        for j in range(4):
            true_theta[j] = float(sys.argv[j + 1])
    generate_data()

    for element in samples:
        if tuple(element[1]) not in samples_by_x:
            samples_by_x[tuple(element[1])] = []
        samples_by_x[tuple(element[1])].append(element[0])

    test_data = []
    train_data = []

    for x in list(itertools.product(X, X, X)):
        if x in samples_by_x:
            train_data.extend([(y, list(x)) for y in samples_by_x[x][2:]])
            test_data.extend([(y, list(x)) for y in samples_by_x[x][0:2]])

    best_theta = train(train_data)

    print
    print "best_theta = " + str(best_theta)
    print "true_theta = " + str(true_theta)
    diff = best_theta - true_theta
    norm_diff = np.linalg.norm(diff)
    print "L2 difference = " + str(norm_diff)

    (
        best_expert_count_correct,
        best_expert_total_count,
        best_expert_accuracy,
        best_expert_total_log_likelihood
    ) = get_accuracy(true_theta, test_data)
    print "best_expert_accuracy = {accuracy} ({count}/{total})".format(
        accuracy=best_expert_accuracy,
        count=best_expert_count_correct,
        total=best_expert_total_count
    )
    print "average best_expert_log_likelihood = {t}".format(t=best_expert_total_log_likelihood/float(best_expert_total_count))

    print
    (
        count_correct,
        total_count,
        accuracy,
        total_log_likelihood
    ) = get_accuracy(best_theta, test_data)
    print "accuracy = {accuracy} ({count}/{total})".format(
        accuracy=accuracy,
        count=count_correct,
        total=total_count
    )
    print "average log_likelihood = {t}".format(t=total_log_likelihood/float(total_count))


def get_accuracy(theta, test_data):
    print "predicting and getting accuracy..."
    count = 0
    count_correct = 0
    total_log_likelihood = 0.0
    for sample in test_data:
        y_correct = sample[0]
        x = sample[1]
        (y, log_likelihood) = predict(x, theta)
        total_log_likelihood += log_likelihood
        correct = False
        if(y == y_correct):
            correct = True
            count_correct += 1
        if(prediction_verbose):
            print "x:{x}, expected: {expected}, predicted: {predicted}, correct?: {correct}".format(
                x=x,
                expected=y_correct,
                predicted=y,
                correct=correct
            )
        count += 1
    if count == 0:
        return (count_correct, count, None, total_log_likelihood)
    return (count_correct, count, float(count_correct)/count, total_log_likelihood)

def predict(x, theta):
    maxValue = -1.0
    maxY = (0, 0, 0)
    for y in list(itertools.product(Y, Y, Y)):
        value = p(y, x, theta)
        if (value > maxValue):
            maxValue = value
            maxY = y
    return (maxY, maxValue)

def train(train_data):
    theta_hat = np.array([0.0, 0.0, 0.0, 0.0])
    theta_hat_average = np.array([0.0, 0.0, 0.0, 0.0])
    total = np.array([0.0, 0.0, 0.0, 0.0])
    counter = 0
    for sample in train_data:
        counter += 1
        if(learning_verbose):
            print counter
        y = sample[0]
        x = sample[1]
        gradient = gradient_phi(y, x)
        Z = calculate_Z(x, theta_hat_average)
        E_grad_phi = expectation_gradient_phi(x, theta_hat, Z)
        new_grad = step_size * (gradient - E_grad_phi)
        theta_hat += new_grad
        # total += theta_hat
        # theta_hat_average = total / counter
        theta_hat_average = (counter - 1)/float(counter)*theta_hat_average + 1/float(counter)*theta_hat
        if(learning_verbose):
            print theta_hat_average

    return theta_hat_average


def expectation_gradient_phi(x, theta, Z):
    total = np.array([0.0, 0.0, 0.0, 0.0])
    ystar = calculate_ystar(x)
    # calculate common factor
    exponent = 0.0
    for j in range(3):
        if(ystar[j] == int(math.ceil(math.log(x[j], 2)))):
            exponent += theta[j]
    common_factor = math.exp(exponent)

    for i in range(3):
        the_sum = 0.0
        product = 1.0
        for j in range(3):
            if j != i:
                product *= (m + math.exp(theta[j]))
            else:
                product *= math.exp(theta[j])
        the_sum += product / Z

        if(ystar[i] == int(math.ceil(math.log(x[i], 2)))):
            the_sum += (math.exp(theta[3]) - 1) * common_factor / Z
        total[i] = the_sum

    total[3] = float(1)/Z * math.exp(theta[3]) * common_factor
    # Before this it was slow because I iterated through all Y^3
    # for y in list(itertools.product(Y, Y, Y)):
    #     gradient = gradient_phi(y, x)
    #     total += p(y, x, theta) * gradient
    return total

def gradient_phi(y, x):
    g = []
    for i in range(3):
        element = 0.0
        if(y[i] == int(math.ceil(math.log(x[i], 2)))):
            element = 1.0
        g.append(element)

    ystar = calculate_ystar(x)
    element = 0.0
    if(list(y) == ystar):
        element = 1.0
    g.append(element)
    g = np.array(g)
    return g

def generate_data():
    print "generating data..."
    for x in list(itertools.product(X, X, X)):
        probability[x] = {}
        Z[x] = calculate_Z(x, true_theta)
        for y in list(itertools.product(Y, Y, Y)):
            probability[x][y] = p(y, x, true_theta)

        # sanity check
        # print sum(probability[x].values())

        for i in range(num_samples_per_x):
            y = sample(x)
            samples.append((y, x))
    print "done generating data. Total of {size} samples".format(size=len(samples))
    # print simplejson.dumps(samples)

def sample(x):
    r = random.uniform(0, 1)
    s = 0
    the_item = (0, 0, 0)
    for item in probability[x]:
        prob = probability[x][item]
        s += prob
        if s >= r:
            the_item = item
            break
    return the_item

def calculate_ystar(x):
    """returns a list of digits of y stars"""
    ystar = sum(x)
    ystar_digits = []
    for i in range(3):
        ystar_digits.append(ystar % 10)
        ystar = ystar / 10
    return ystar_digits

def calculate_Z(x, theta):
    ystar = calculate_ystar(x)
    total = 0.0
    product = 1.0
    for i in range(3):
        product *= (m + math.exp(theta[i]))
    total += product

    exponent = 0.0
    for i in range(3):
        if(ystar[i] == int(math.ceil(math.log(x[i], 2)))):
            exponent += theta[i]
    total += (math.exp(theta[3]) - 1) * math.exp(exponent)
    return total

def p(y, x, theta):
    exponent = 0.0
    ystar = calculate_ystar(x)
    for i in range(3):
        if(y[i] == int(math.ceil(math.log(x[i], 2)))):
            exponent += theta[i]
    if(list(y) == ystar):
        exponent += theta[3]
    return math.exp(exponent)/calculate_Z(x, theta)


if __name__ == '__main__':
    main()