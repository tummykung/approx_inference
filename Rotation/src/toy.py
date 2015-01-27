#!/usr/bin/python
# usage: python toy.py theta0 theta1 theta2 theta3
import sys
import itertools
import math
import random
import simplejson
import numpy as np

k = 3 # sentence length
n = 4 # range X

range_Y = int(np.ceil(np.log(n)/np.log(2))) + 1
m = [range_Y, range_Y, 9]

learning_verbose = False
prediction_verbose = False
sanity_check = False
true_theta = np.array([0.0, 0.0, 0.0, 0.0]) # to be initialized
X = range(1, n + 1)
Y0 = range(m[0] + 1)
Y1 = range(m[1] + 1)
Y2 = range(m[2] + 1)
W = range(3)
Z = {}
probability = {}
num_samples_per_x = 15
samples = []
samples_by_x = {}
eta = None # eta = None means we use a non-relaxed model

step_size = 0.1

def main():
    eta = None # this is needed.
    if(len(sys.argv) < 5 or len(sys.argv) > 6):
        print "(current number of argumets = {num}".format(num=len(sys.argv) - 1)
        print "Must have at least 4 arguments for the 4 values of theta."
        print "If the fifth value is specified, that will be eta, and we'll use a relaxed model with"
        print "the corresponding eta"
        sys.exit(2)
    else:
        for j in range(4):
            true_theta[j] = float(sys.argv[j + 1])
        if (len(sys.argv) == 6):
            eta = float(sys.argv[5])
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
    if eta is not None:
        print "eta = " + str(eta)
    else:
        print "no relaxation"

    print "rangeY = " + str(m)
    print "vocab size(X) = " + str(n)

    print "best_theta = " + str(best_theta)
    print "true_theta = " + str(true_theta)
    diff = best_theta - true_theta
    norm_diff = np.linalg.norm(diff)
    print "L2 difference = " + str(norm_diff)
    print

    (
        best_expert_count_correct,
        best_expert_total_count,
        best_expert_accuracy,
        best_expert_total_log_likelihood
    ) = get_accuracy(true_theta, test_data, eta)
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
    ) = get_accuracy(best_theta, test_data, eta)
    print "accuracy = {accuracy} ({count}/{total})".format(
        accuracy=accuracy,
        count=count_correct,
        total=total_count
    )
    print "average log_likelihood = {t}".format(t=total_log_likelihood/float(total_count))


def get_accuracy(theta, test_data, eta):
    print "predicting and getting accuracy..."
    count = 0
    count_correct = 0
    total_log_likelihood = 0.0
    for sample in test_data:
        w_correct = sample[0]
        x = sample[1]
        (w, log_likelihood) = predict(x, theta, eta)
        total_log_likelihood += log_likelihood
        correct = False
        if(w == w_correct):
            correct = True
            count_correct += 1
        if(prediction_verbose):
            print "x:{x}, expected: {expected}, predicted: {predicted}, correct?: {correct}".format(
                x=x,
                expected=w_correct,
                predicted=w,
                correct=correct
            )
        count += 1
    if count == 0:
        return (count_correct, count, None, total_log_likelihood)
    return (count_correct, count, float(count_correct)/count, total_log_likelihood)

def predict(x, theta, eta):
    maxValue = -1.0
    maxW = (0, 0, 0)
    for w in itertools.product(W, W, W):
        value = real_p(w, x, theta, eta)
        if (value > maxValue):
            maxValue = value
            maxW = w

    return (maxW, maxValue)

def train(train_data):
    theta_hat = np.array([0.0, 0.0, 0.0, 0.0])
    theta_hat_average = np.array([0.0, 0.0, 0.0, 0.0])
    counter = 0
    for sample in train_data:
        counter += 1
        if(learning_verbose):
            print counter
        w = sample[0]
        x = sample[1]
        new_grad = np.array([0.0, 0.0, 0.0, 0.0])
        for y in itertools.product(Y0, Y1, Y2):
            y = np.array(y)
            gradient = gradient_phi(y, x)
            Z = calculate_Z(x, theta_hat_average)
            E_grad_phi = expectation_gradient_phi(x, theta_hat, Z)
            grad_log_Y = step_size * (gradient - E_grad_phi)

            if eta is None:
                the_q_component = q(w, y)
            else:
                the_q_component = q_relaxed(w, y, eta)
            new_grad += grad_log_Y * p(y, x, theta_hat) * the_q_component

        new_grad /= real_p(w, x, theta_hat, eta)

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
                product *= (m[j] + math.exp(theta[j]))
            else:
                product *= math.exp(theta[j])
        the_sum += product / Z

        if(ystar[i] == int(math.ceil(math.log(x[i], 2)))):
            the_sum += (math.exp(theta[3]) - 1) * common_factor / Z
        total[i] = the_sum

    total[3] = float(1)/Z * math.exp(theta[3]) * common_factor
    # Before this it was slow because I iterated through all Y^3
    # for y in list(itertools.product(Y0, Y1, Y2)):
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
    for x in itertools.product(X, X, X):
        # can't convert to tuple since can't hash array
        probability[x] = {}
        Z[x] = calculate_Z(x, true_theta)
        for w in list(itertools.product(W, W, W)):
            probability[x][w] = real_p(w, x, true_theta, None) # important to not generate from eta

        # sanity check
        if sanity_check:
            print "sanity check... " + str((x, sum(probability[x].values())))

        for i in range(num_samples_per_x):
            w = sample(x)
            samples.append((w, x))
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
        product *= (m[i] + math.exp(theta[i]))
    total += product

    exponent = 0.0
    for i in range(3):
        if(ystar[i] == int(math.ceil(math.log(x[i], 2)))):
            exponent += theta[i]
    total += (math.exp(theta[3]) - 1) * math.exp(exponent)
    return total

def real_p(w, x, theta, eta):
    total = 0.0
    for y in itertools.product(Y0, Y1, Y2):
        y = np.array(y)
        if eta is None:
            q_factor = q(w, y)
        else:
            q_factor = q_relaxed(w, y, eta)
        total += p(y, x, theta) * q_factor
    return total

def q_relaxed(w, y, eta):
    return np.exp(-eta * hamming(w, denotation(y)))/np.power(1.0 + 2.0 * np.exp(-eta), 3.0)

def hamming(x, y):
    """x and y must have the same length, else it will cut off the tail of the longer one."""
    total = 0
    min_length = np.min([len(x), len(y)])
    for i in range(min_length):
        total += np.abs(x[i] - y[i])
    return total

def denotation(y):
    return y % 3 # assuming y is np.array, so this operator should work element-wise

def q(w, y):
    output = True
    for j in range(3):
        if (y[j] % 3 != w[j]):
            output = False
    if output:
        return 1.0
    else:
        return 0.0

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