
class CRFModel:
    def __init__(
        self,
        num_layers,
        arities,
        ranges,
        dim_theta,
        potentials,
        learning_verbose = False,
        state_verbose = False,
        log_likelihood_verbose = True,
        prediction_verbose = False,
        sanity_check = False,
        fully_supervised = False
    ):
        """
        :param num_layers: an int representing a number of layers in the CRF graph
        :param arities: a list of arities of each layer
        :param ranges: a list of a list of ranges. One list per one layer
        """
        self.num_layers = num_layers
        self.arities = arities
        self.ranges = ranges
        self.dim_theta = dim_theta
        self.potentials = potentials
        self.learning_verbose = learning_verbose
        self.state_verbose = state_verbose
        self.log_likelihood_verbose = log_likelihood_verbose
        self.prediction_verbose = prediction_verbose
        self.sanity_check = sanity_check
        self.fully_supervised = fully_supervised
