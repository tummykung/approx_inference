from util.probability import ConditionalSampling

def generate_data(model):
    """
    :param model: CRFModel instance
    """
    # if state_verbose:
    #     print >> sys.stderr, "generating data..."
    for x in itertools.product(X, X, X):
        # can't convert to tuple since can't hash array
        probability[x] = {}
        Z[x] = calculate_Z(x, true_theta)

        if fully_supervised:
            for y in itertools.product(Y0, Y1, Y2):
                probability[x][y] = p(y, x, true_theta)
        else:
            for w in list(itertools.product(W, W, W)):
                probability[x][w] = real_p(w, x, true_theta, None) # important to not generate from xi
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