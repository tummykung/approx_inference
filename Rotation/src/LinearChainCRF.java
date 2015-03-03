import java.util.Arrays;
import java.util.Random;

import util.CartesianProduct;
import fig.basic.Option;
import fig.prob.SampleUtils;

public class LinearChainCRF {
  // input:
  double[][] edgeWeights, nodeWeights; // according to a HMM model
  private int approx_inference = 0;
  private int M; // the number of iterations in approximation inference
  //  Inference Type
  //  ==============
  //  0 = Exact
  //  1 = Approximate inference by
  //    - only sampling the model expectation
  //  2 = Importance sampling by
  //    - sampling the model expectation, using the importance weight
  //      to correct target sampling. However, still use exact inference
  //      to compute the importance weight.
  //  3 = Normalized importance sampling by
  //    - using the model weight as the proposal distribution
  private double xi; // relaxation parameter
  private double eta0 = 0.01; // gradient descent initial stepsize
  private int gradient_descent_type = 0;
  // 0 = constant step size
  // 1 = decreasing stepsize
  @Option(required=true)
  private boolean fully_supervised = true; // can switch to fully supervised for sanity check
  @Option(required=false)
  private boolean learning_verbose = true;
  @Option(required=false)
  private boolean state_verbose = true;
  @Option(required=false)
  private boolean log_likelihood_verbose = true;
  @Option(required=false)
  private boolean prediction_verbose = false;
  @Option(required=false)
  private boolean sanity_check = false;

  // internal
  private long seed;
  private int d; // dimension of the parameters
  private int numX;
  private int numY;
  private int[][][] data;
  private Random randomizer;
  private double eta; // stepsize
  ForwardBackward my_model;
  
  
  public LinearChainCRF(
      long seed,
      double[][] edgeWeights,
      double[][] nodeWeights,
      double xi,
      int approx_inference,
      int M
    ) {
    this.seed = seed;
    this.edgeWeights = edgeWeights; // Y x Y
    this.nodeWeights = nodeWeights; // Y x X
    this.xi = xi;
    this.approx_inference = 0;
    this.M = M;
    this.numX = edgeWeights.length;
    assert(edgeWeights.length == nodeWeights.length);
    this.numY = nodeWeights[0].length;
    this.d = this.numY * (this.numX + this.numY);

    randomizer = new Random(seed);
    my_model = new ForwardBackward(edgeWeights, nodeWeights);
  }

  public int[][][] generate_data(int num_samples) {
    // return[n][0] = x is int[] and return[n][1] = y is also int[]
    // and n is the num sample iterator

    data = new int[num_samples][2][nodeWeights.length];
    // given parameters, generate data according to a log-linear model
    // fully supervised:
    // p_{theta}(y | x) = 1/Z(theta; x) * exp(theta' * phi(x, z))
    //
    // full model:
    // p_{theta, xi} (y, z | x) = q_{xi}(y | z) * p_{theta}(z | x)
    // p_{theta}(z | x) = 1/Zp(theta; x) * exp(theta' * phi(x, z))
    // q_{xi}(y | z) = 1/Zq(xi; z) * exp(xi' * psi(y, z))

    if(state_verbose) {
      System.out.println("generating data...");
    }
    my_model.infer();
    for (int i = 0; i < num_samples; i++) {
      int[] seq = my_model.sample(randomizer);
      data[i][0] = seq;
      // TODO: change weight based on x. Right now just do identical.
      data[i][1] = seq;
    }

    if(state_verbose) {
      System.out.println("done genetaing data.\nnum data instances: " + data.length);
      if(data.length < 100) {
        System.out.println("data: " + Arrays.deepToString(data));
      }
    }
    return data;
  }
  
  public double[] train(int[][][] train_data) {
    // returns: learned_theta
    double[] theta_hat = new double[d];
    double[] theta_hat_average = new double[d];
    int counter = 0;
    for (int[][] sample : train_data) {
      counter += 1;
      int[] y = sample[0];
      int[] z = sample[0]; // different semantics
      int[] x = sample[1];
      double[] new_grad = new double[d];
      
      
      if (fully_supervised) {
        double[] gradient = phi(z, x);
//        Z = calculate_Z(x, theta_hat)
//        E_grad_phi = expectation_phi(x, theta_hat, Z, approx_inference) // old version
          double[] E_grad_phi = expectation_phi(x, theta_hat);
          double[] grad_log_Z = new double[d];
          for(int i = 0; i< d; i++)
            grad_log_Z[i] = gradient[i] - E_grad_phi[i];
          new_grad = grad_log_Z;
      } else {
//        Z = calculate_Z(x, theta_hat)
//        if approx_inference == 1:
//          E_grad_phi = expectation_phi(x, theta_hat, Z, approx_inference)
//          for z in itertools.product(Z0, Z1, Z2):
//            z = np.array(z)
//            gradient = phi(z, x)
//            grad_log_Z = (gradient - E_grad_phi)
//
//            if xi is None:
//              the_q_component = q(y, z)
//            else:
//              the_q_component = q_relaxed(y, z, xi)
//            new_grad += grad_log_Z * p(z, x, theta_hat) * the_q_component
//
//          new_grad /= real_p(y, x, theta_hat, xi)
//        elif approx_inference == 2:
//          new_grad = difference_between_expectations(x, y, theta_hat, Z, xi, M)
      }

      if(gradient_descent_type == Main.DECREASING_STEP_SIZES) {
        eta = eta0 * 1.00/Math.sqrt(counter);
      } else if (gradient_descent_type == Main.CONSTANT_STEP_SIZES){
        eta = eta0;
      }

      for(int i = 0; i < d; i++) {
        theta_hat[i] += eta * new_grad[i];
        theta_hat_average[i] =
            (counter - 1)/(double)counter*theta_hat_average[i] +
            1/(double)counter*theta_hat[i];
      }
      
      if((log_likelihood_verbose && counter % 100 == 0) || (counter == train_data.length)) {
        double average_log_likelihood = calculate_average_log_likelihood(train_data, theta_hat_average);
        System.out.println(
            counter + 
            ": train dataset average log-likelihood:\t" +
            average_log_likelihood
        );
      }
      if(learning_verbose) {
        System.out.println(counter + ": theta_hat_average:\t" + Arrays.toString(theta_hat_average));
      }

    }
    return theta_hat_average;
  }

  private double[] expectation_phi(int[] x, double[] theta) {
    double[] total = new double[d];
    int[] indicesForZ = new int[numY];
    for(int i = 0; i < numY; i++) {
      indicesForZ[i] = numY; //TODO: check this
    }
    if(approx_inference == Main.EXACT) {
      for(int[] z : new CartesianProduct(indicesForZ)) {
        double[] the_phi = phi(z, x);
        for (int k = 0; k < d; k++)
          total[k] += p(z, x, theta) * the_phi[k];
      }
    }
    return total;
  }

  private int indexTransition(int i, int j) {
    assert(0 <= i && i < numY);
    assert(0 <= j && j < numY);
    return numY*i + j;
  }

  private int indexEmission(int i, int j) {
    assert(0 <= i && i < numY);
    assert(0 <= j && j < numX);
    return numY*numY + numX*i + j;
  }

  private double[] phi(int[] z, int[] x) {
    double[] output = new double[d];
    for (int aIndex = 0; aIndex < numY; aIndex++) {
      for (int cIndex = 0; cIndex < numX; cIndex++) {
        output[indexEmission(aIndex, cIndex)] = 1;
      }
    }

    for (int aIndex = 0; aIndex < numY; aIndex++) {
      for (int bIndex = 0; bIndex < numY; bIndex++) {
        output[indexTransition(aIndex, bIndex)] = 1;
      }
    }
    return output;
  }

  public double calculate_average_log_likelihood(int[][][] train_data, double theta[]) {
    double total_log_likelihood = 0.0;
    int count = 0;
    for (int[][] sample : train_data) {
      count += 1;
      int[] y = sample[0];
      int[] z = sample[0];
      int[] x = sample[1];
      if(fully_supervised) {
        total_log_likelihood += Math.log(p(z, x, theta));
      } else {
        total_log_likelihood += Math.log(real_p(y, x, theta, xi));
      }
    }
    return total_log_likelihood/(double)count;
  }
  public double difference_between_expectations(
      int[] x, int[] y, double[] theta, double Z) {
//    TODO
//    output = np.array([0.0, 0.0, 0.0, 0.0])
//    total1 = np.array([0.0, 0.0, 0.0, 0.0])
//    totalw = 0.0
//    total2 = np.array([0.0, 0.0, 0.0, 0.0])
//    # construct the probability
//    the_probability = {}
//    for x in itertools.product(X, X, X):
//        # can't convert to tuple since can't hash array
//        the_probability[x] = {}
//
//        for z in itertools.product(Z0, Z1, Z2):
//            the_probability[x][z] = p(z, x, theta)
//
//    sampler = ConditionalSampling(the_probability)
//
//    for i in range(M):
//        z = sampler.sample(x)
//        z = np.array(z)
//        # print "z: " + str(z)
//        w = q_relaxed(y, z, xi)
//        # print "w: " + str(w)
//        the_phi = phi(z, x)
//        # print "the_phi: " + str(the_phi)
//        totalw += w
//        total1 += w * the_phi
//        total2 += the_phi
//
//    try:
//        output = total1/totalw - total2/M
//    except:
//        output = np.array([0.0, 0.0, 0.0, 0.0])
//
//    return output
    return 0.0;
  }
  public double getLogZ() {
    return my_model.getLogZ();
  }
  public double real_p(int[] y, int[] x, double[] theta, double xi) {
    // TODO
    return 0.5;
  }
    
  public double p(int[] z, int[] x, double[] theta) {
    // TODO
    return 1;
  }
}
