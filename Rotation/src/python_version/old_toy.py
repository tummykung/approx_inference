import itertools
import math
import random
import simplejson

k = 3
n = 20
m = 9
data = "data.json"

true_theta = [0.2, 0.5, 0.1, 6]
X = range(1, n + 1)
Y = range(m + 1)
Z = {}
probability = {}
num_samples_per_x = 15
samples = []
samples_by_x = {}

step_size = 0.0
import itertools
import math
import random
import simplejson

k = 3
n = 20
m = 9
data = "data.json"

true_theta = [0.2, 0.5, 0.1, 6]
X = range(1, n + 1)
Y = range(m + 1)
Z = {}
probability = {}
num_samples_per_x = 15
samples = []
samples_by_x = {}

step_size = 0.01

def main():
    # generate_data()
    with open(data) as f:
        samples_temp = simplejson.load(f)

    for element in samples_temp:
        samples.append(element)

        if tuple(element[1]) not in samples_by_x:
            samples_by_x[tuple(element[1])] = []
        samples_by_x[tuple(element[1])].append(element[0])

    test_data = []
    train_data = []

    for x in list(itertools.product(X, X, X)):
        train_data.extend([(y, list(x)) for y in samples_by_x[x][0:3]])
        test_data.extend([(y, list(x)) for y in samples_by_x[x][3:4]])

    best_theta = train(train_data)

    print best_theta
    print get_accuracy(best_theta, test_data)

def get_accuracy(best_theta, test_data):
    count = 0
    count_correct = 0
    for sample in test_data:
        y_correct = sample[0]
        x = sample[1]
        y = predict(x, theta)
        if(y == y_correct):
            count_correct += 1
        count += 1
    return (float)(count_correct)/count

def predict(x, theta):
    maxValue = 0.0
    maxY = (0, 0, 0)
    for y in list(itertools.product(Y, Y, Y)):
        value = p(y, x, theta)
        if (value > maxValue):
            maxValue = value
            maxY = y
    return y

def train(train_data):
    theta_hat = [0, 0, 0, 0]
    counter = 0
    for sample in train_data:
        counter += 1
        print counter
        y = sample[0]
        x = sample[1]
        gradient = gradient_phi(y, x)
        E_grad_phi = expectation_gradient_phi(x, theta_hat)
        for i in range(4):
            theta_hat[i] += step_size * (gradient[i] - E_grad_phi[i])
        print theta_hat

    return theta_hat


def expectation_gradient_phi(x, theta):
    total = [0, 0, 0, 0]
    for y in list(itertools.product(Y, Y, Y)):
        gradient = gradient_phi(y, x)
        for i in range(4):
            total[i] += p(y, x, theta) * gradient[i]
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
    return g

def generate_data():
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

    print simplejson.dumps(samples)

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

def main():
    # generate_data()
    with open(data) as f:
        samples_temp = simplejson.load(f)

    for element in samples_temp:
        samples.append(element)
        samples_by_x[element[0]] = element[1]

    import ipdb; ipdb.set_trace()
    best_theta = train()
    print best_theta

def train():
    theta_hat = [0, 0, 0, 0]
    counter = 0
    for sample in samples:
        counter += 1
        print counter
        y = sample[0]
        x = sample[1]
        gradient = gradient_phi(y, x)
        E_grad_phi = expectation_gradient_phi(x, theta_hat)
        for i in range(4):
            theta_hat[i] += step_size * (gradient[i] - E_grad_phi[i])
        print theta_hat

    return theta_hat


def expectation_gradient_phi(x, theta):
    total = [0, 0, 0, 0]
    for y in list(itertools.product(Y, Y, Y)):
        gradient = gradient_phi(y, x)
        for i in range(4):
            total[i] += p(y, x, theta) * gradient[i]
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
    return g

def generate_data():
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

    print simplejson.dumps(samples)

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