#!/usr/bin/python
import sys
import itertools
import math
import random
import argparse
import numpy as np
from util.probability import ConditionalSampling
from util.probability import indicator
from util.probability import hamming

from model import CRFModel


k = 3 # sentence length
n = 5 # range X

range_Z = int(np.ceil(np.log(n)/np.log(2))) + 1
# m = [range_Z, range_Z, 9]
m = [3, 3, 9]

learning_verbose = False
state_verbose = False
log_likelihood_verbose = True
prediction_verbose = False
sanity_check = False
fully_supervised = False
approx_inference = True
true_theta = np.array([0.0, 0.0, 0.0, 0.0]) # to be initialized
X = range(1, n + 1)
Z0 = range(m[0] + 1)
Z1 = range(m[1] + 1)
Z2 = range(m[2] + 1)
Y = range(3)
Z = {}
probability = {}
num_samples_per_x = 10
samples = []
samples_by_x = {}
xi = None # xi = None means we use a non-relaxed model

step_size = 0.1

# the_model = CRFModel(
#     num_layers=2,
#     arities=[3,3],
#     ranges=[[X, X, X], [Z0, Z1, Z2]],
#     dim_theta=5, # theta0, theta1, theta2, theta3, xi
#     potentials= [
#         lambda x,y,z: indicator(z[0] == int(math.ceil(math.log(x[0], 2)))),
#         lambda x,y,z: indicator(z[1] == int(math.ceil(math.log(x[1], 2)))),
#         lambda x,y,z: indicator(z[2] == int(math.ceil(math.log(x[2], 2)))),
#         lambda x,y,z:
#             indicator(z[0] == int(math.ceil(math.log(x[0], 2)))),
#         lambda x,y,z: -
#     ]
#     learning_verbose = False
#     state_verbose = False
#     log_likelihood_verbose = True
#     prediction_verbose = False
#     sanity_check = False
#     fully_supervised = False
# )

# import ipdb; ipdb.set_trace()


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-f', '--f', type=bool, help="fully supervised", default=False)
    # args = parser.parse_args()
    # import ipdb; ipdb.set_trace()
    # fully_supervised = args.f

    xi = None # this is needed.
    if(len(sys.argv) < 6 or len(sys.argv) > 7):
        print "(current number of argumets = {num}".format(num=len(sys.argv) - 1)
        print "Must have at least 4 arguments for the 4 values of theta."
        print "If the fifth value is specified, that will be xi,"
        print "and we'll use a relaxed model with parameter xi"
        sys.exit(2)
    else:
        if (sys.argv[1] == "full"):
            fully_supervised = True
        elif (sys.argv[1] == "mod3"):
            fully_supervised = False
        else:
            print "the first argument must be either 'full' or 'mod3'"
            sys.exit(2)
        for j in range(4):
            true_theta[j] = float(sys.argv[j + 2])
        if (len(sys.argv) == 7):
            xi = float(sys.argv[6])

    generate_data()

    for element in samples:
        if tuple(element[1]) not in samples_by_x:
            samples_by_x[tuple(element[1])] = []
        samples_by_x[tuple(element[1])].append(element[0])

    test_data = []
    train_data = []

    for x in list(itertools.product(X, X, X)):
        if x in samples_by_x:
            train_data.extend([(output, list(x)) for output in samples_by_x[x][2:]])
            test_data.extend([(output, list(x)) for output in samples_by_x[x][0:2]])

    random.shuffle(train_data)
    learned_theta = train(train_data, xi)

    print
    if xi is not None:
        print "xi:\t" + str(xi)
    else:
        print "xi:\tno relaxation"

    print "rangeY:\t" + str(m)
    print "vocab size(X):\t" + str(n)
    print "fully_supervised?:\t" + str(fully_supervised)

    print

    print "===== expert statistics ====="
    print "true_theta:\t" + str(true_theta)
    best_expert_train_dataset_average_log_likelihood = calculate_average_log_likelihood(train_data, true_theta, xi)
    print "best_expert_train_dataset_average_log_likelihood:\t" + str(best_expert_train_dataset_average_log_likelihood)
    (
        best_expert_count_correct,
        best_expert_total_count,
        best_expert_accuracy,
        best_expert_test_dataset_average_log_likelihood
    ) = get_accuracy(true_theta, test_data, xi)
    print "best_expert_test_dataset_average_log_likelihood:\t{ll}".format(ll=best_expert_test_dataset_average_log_likelihood)
    print "best_expert_accuracy:\t{accuracy} ({count}/{total})".format(
        accuracy=best_expert_accuracy,
        count=best_expert_count_correct,
        total=best_expert_total_count
    )

    print

    print "===== learner statistics ====="
    print "learned_theta:\t" + str(learned_theta)
    learner_train_dataset_average_log_likelihood = calculate_average_log_likelihood(train_data, learned_theta, xi)
    print "learner_train_dataset_average_log_likelihood:\t" + str(learner_train_dataset_average_log_likelihood)
    (
        count_correct,
        total_count,
        accuracy,
        learner_test_dataset_average_log_likelihood
    ) = get_accuracy(learned_theta, test_data, xi)
    print "learner_test_dataset_average_log_likelihood:\t{ll}".format(ll=learner_test_dataset_average_log_likelihood)
    print "learner_accuracy:\t{accuracy} ({count}/{total})".format(
        accuracy=accuracy,
        count=count_correct,
        total=total_count
    )
    # diff = learned_theta - true_theta
    # norm_diff = np.linalg.norm(diff)
    # print "L2 difference:\t" + str(norm_diff)

def calculate_average_log_likelihood(dataset, theta, xi):
    total_log_likelihood = 0.0
    count = 0
    for sample in dataset:
        count += 1
        y = sample[0]
        z = sample[0] # different semantics
        x = sample[1]
        if fully_supervised:
            total_log_likelihood += np.log(p(z, x, theta))
        else:
            total_log_likelihood += np.log(real_p(y, x, theta, xi))
    return total_log_likelihood/float(count)


def get_accuracy(theta, test_data, xi):
    if state_verbose:
        print >> sys.stderr, "predicting and getting accuracy..."
    count = 0
    count_correct = 0
    for sample in test_data:
        output_correct = sample[0]
        x = sample[1]
        (output, maxProbablity) = predict(x, theta, xi)
        correct = False
        if(output == output_correct):
            correct = True
            count_correct += 1
        if(prediction_verbose):
            print "x:{x}, expected: {expected}, predicted: {predicted}, correct?: {correct}".format(
                x=x,
                expected=output_correct,
                predicted=output,
                correct=correct
            )
        count += 1
    average_log_likelihood = calculate_average_log_likelihood(test_data, theta, xi)
    print "test dataset average log-likelihood:\t" + str(average_log_likelihood)
    if count == 0:
        return (count_correct, count, None, average_log_likelihood)
    return (count_correct, count, float(count_correct)/count, average_log_likelihood)

def predict(x, theta, xi):
    maxValue = -1.0
    maxOutput = (0, 0, 0)
    if fully_supervised:
        for z in itertools.product(Z0, Z1, Z2):
            value = p(z, x, theta)
            if (value > maxValue):
                maxValue = value
                maxOutput = z
    else:
        for y in itertools.product(Y, Y, Y):
            value = real_p(y, x, theta, xi)
            if (value > maxValue):
                maxValue = value
                maxOutput = y

    return (maxOutput, maxValue)

def train(train_data, xi):
    theta_hat = np.array([0.0, 0.0, 0.0, 0.0])
    theta_hat_average = np.array([0.0, 0.0, 0.0, 0.0])
    counter = 0
    for sample in train_data:
        counter += 1
        y = sample[0]
        z = sample[0] # different semantics
        x = sample[1]
        new_grad = np.array([0.0, 0.0, 0.0, 0.0])

        if fully_supervised:
            z = np.array(z)
            gradient = gradient_phi(z, x)
            Z = calculate_Z(x, theta_hat)
            E_grad_phi = expectation_gradient_phi(x, theta_hat, Z)
            grad_log_Z = (gradient - E_grad_phi)
            new_grad = grad_log_Z

        else:
            Z = calculate_Z(x, theta_hat)
            E_grad_phi = expectation_gradient_phi(x, theta_hat, Z)
            for z in itertools.product(Z0, Z1, Z2):
                z = np.array(z)
                gradient = gradient_phi(z, x)
                grad_log_Z = (gradient - E_grad_phi)

                if xi is None:
                    the_q_component = q(y, z)
                else:
                    the_q_component = q_relaxed(y, z, xi)
                new_grad += grad_log_Z * p(z, x, theta_hat) * the_q_component

            new_grad /= real_p(y, x, theta_hat, xi)

        theta_hat += step_size * new_grad
        # total += theta_hat
        # theta_hat_average = total / counter
        theta_hat_average = (counter - 1)/float(counter)*theta_hat_average + 1/float(counter)*theta_hat

        if (log_likelihood_verbose and counter % 100 == 0) or (counter == len(train_data)):
            average_log_likelihood = calculate_average_log_likelihood(train_data, theta_hat_average, xi)
            print "{counter}: train dataset average log-likelihood:\t{ll}".format(
                counter=counter,
                ll=average_log_likelihood
            )
        if(learning_verbose):
            print "{counter}: theta_hat_average:\t{theta_hat_average}".format(
                counter=counter,
                theta_hat_average=theta_hat_average
            )

    return theta_hat_average


def expectation_gradient_phi(x, theta, Z):
    total = np.array([0.0, 0.0, 0.0, 0.0])
    zstar = calculate_zstar(x)
    # calculate common factor
    exponent = 0.0
    for j in range(3):
        if(zstar[j] == int(math.ceil(math.log(x[j], 2)))):
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

        if(zstar[i] == int(math.ceil(math.log(x[i], 2)))):
            the_sum += (math.exp(theta[3]) - 1) * common_factor / Z
        total[i] = the_sum

    total[3] = float(1)/Z * math.exp(theta[3]) * common_factor
    # sanity check
    # Before this it was slow because I iterated through all Z^3
    if sanity_check:
        new_total = 0.0
        for z in list(itertools.product(Z0, Z1, Z2)):
            gradient = gradient_phi(z, x)
            new_total += p(z, x, theta) * gradient
        print "if expectation_gradient_phi is correct, we should have {a}={b}".format(
            a=total,
            b=new_total
        )
    return total

def gradient_phi(z, x):
    g = []
    for i in range(3):
        element = 0.0
        if(z[i] == int(math.ceil(math.log(x[i], 2)))):
            element = 1.0
        g.append(element)

    zstar = calculate_zstar(x)
    element = 0.0
    if(list(z) == zstar):
        element = 1.0
    g.append(element)
    g = np.array(g)
    return g

def generate_data():
    if state_verbose:
        print >> sys.stderr, "generating data..."
    for x in itertools.product(X, X, X):
        # can't convert to tuple since can't hash array
        probability[x] = {}
        Z[x] = calculate_Z(x, true_theta)

        if fully_supervised:
            for z in itertools.product(Z0, Z1, Z2):
                probability[x][z] = p(z, x, true_theta)
        else:
            for y in list(itertools.product(Y, Y, Y)):
                probability[x][y] = real_p(y, x, true_theta, None) # important to not generate from xi
        # sanity check
        if sanity_check and state_verbose:
            print >> sys.stderr, "sanity check... " + str((x, sum(probability[x].values())))

        sampler = ConditionalSampling(probability)
        for i in range(num_samples_per_x):
            output = sampler.sample(x)
            samples.append((output, x))
    if state_verbose:
        print >> sys.stderr, "done generating data."
    print "num_samples:\t{size}".format(size=len(samples))
    # print simplejson.dumps(samples)

def calculate_zstar(x):
    """returns a list of digits of y stars"""
    zstar = sum(x)
    zstar_digits = []
    for i in range(3):
        zstar_digits.append(zstar % 10)
        zstar = zstar / 10
    return zstar_digits

def calculate_Z(x, theta):
    zstar = calculate_zstar(x)
    total = 0.0
    product = 1.0
    for i in range(3):
        product *= (m[i] + math.exp(theta[i]))
    total += product

    exponent = 0.0
    for i in range(3):
        if(zstar[i] == int(math.ceil(math.log(x[i], 2)))):
            exponent += theta[i]
    total += (math.exp(theta[3]) - 1) * math.exp(exponent)
    return total

def real_p(y, x, theta, xi):
    total = 0.0
    for z in itertools.product(Z0, Z1, Z2):
        z = np.array(z)
        if xi is None:
            q_factor = q(y, z)
        else:
            q_factor = q_relaxed(y, z, xi)
        total += p(z, x, theta) * q_factor
    return total

def q_relaxed(y, z, xi):
    return np.exp(-xi * hamming(y, denotation(z)))/np.power(1.0 + 2.0 * np.exp(-xi), 3.0)

def denotation(z):
    return y % 3 # assuming y is np.array, so this operator should work element-wise

def q(y, z):
    output = True
    for j in range(3):
        if (z[j] % 3 != y[j]):
            output = False
    if output:
        return 1.0
    else:
        return 0.0

def p(z, x, theta):
    exponent = 0.0
    zstar = calculate_zstar(x)
    for i in range(3):
        if(z[i] == int(math.ceil(math.log(x[i], 2)))):
            exponent += theta[i]
    if(list(z) == zstar):
        exponent += theta[3]
    return math.exp(exponent)/calculate_Z(x, theta)


if __name__ == '__main__':
    main()