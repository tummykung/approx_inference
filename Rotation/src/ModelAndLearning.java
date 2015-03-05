import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import util.CartesianProduct;
import fig.basic.Option;
import fig.basic.LogInfo;

public class ModelAndLearning {
  private ArrayList<Example> data;
  LinearChainCRF the_model_for_each_x;
  private Random randomizer;

  
  // 0 = constant step size
  // 1 = decreasing step size
 
  private int d; // dimension of the parameters
  private double eta; // stepsize
  
  public ModelAndLearning() {
    // N = Length of sequence (sentenceLength)
    // S = Number of states

    // elements selected from Y times Y 
    // dimension is S x S
    double [][] edgeWeights = new double[][]{
        {1, 10, 1},
        {3, 4, 1},
        {1, 2, 3}
     };

    // elements selected from X times Y
    // dimension is N x S
    double [][] nodeWeights = new double[][]{
        {10, 2, 3},   // at x0
        {4, 10, 4},   // at x1
        {4, 3, 20},   // at x2
        {5, 2, 1}     // at x3
    };
    randomizer = new Random(Main.seed);
  }

  public ArrayList<Example> generate_data(int num_samples, int sentenceLength) {
    // return[n][0] = x is int[] and return[n][1] = y is also int[]
    // and n is the num sample iterator

    data = new ArrayList<Example>();
    // int[num_samples][2][sentenceLength];
    // given parameters, generate data according to a log-linear model
    // fully supervised:
    // p_{theta}(y | x) = 1/Z(theta; x) * exp(theta' * phi(x, z))
    //
    // full model:
    // p_{theta, xi} (y, z | x) = q_{xi}(y | z) * p_{theta}(z | x)
    // p_{theta}(z | x) = 1/Zp(theta; x) * exp(theta' * phi(x, z))
    // q_{xi}(y | z) = 1/Zq(xi; z) * exp(xi' * psi(y, z))

    if(Main.state_verbose) {
      System.out.println("generating data...");
    }

    // TODO: generate this from some distribution. e.g., a uniform distribution
    double [][] edgeWeights = new double[][]{
        {1, 10, 1},
        {3, 4, 1},
        {1, 2, 3}
     };

    // elements selected from X times Y
    // dimension is N x S
    double [][] nodeWeights = new double[][]{
        {10, 2, 3},   // at x0
        {4, 10, 4},   // at x1
        {4, 3, 20},   // at x2
        {5, 2, 1}     // at x3
    };
    LinearChainCRF the_model_for_each_x = new LinearChainCRF(
        edgeWeights,
        nodeWeights
    );
    the_model_for_each_x.inferState();

    for (int i = 0; i < num_samples; i++) {
       int[] seq = the_model_for_each_x.sample(randomizer);
      Example example = new Example(seq, seq);
      // TODO: change weight based on x. Right now just do identical
      data.add(example);
    }

    if(Main.state_verbose) {
      System.out.println("done genetaing data.\nnum data instances: " + data.size());
      if(data.size() < 100) {
        System.out.println("data: " + data);
      }
    }
    return data;
  }


  public double[] train(ArrayList<Example> train_data) {
    LogInfo.begin_track("train");
    // returns: learned_theta
    double[] theta_hat = new double[d];
    double[] theta_hat_average = new double[d];
    int counter = 0;
    for (Example sample : train_data) {
      counter += 1;
      int[] y = sample.getOutput();
      int[] z = sample.getOutput(); // different semantics
      int[] x = sample.getInput();
      double[] new_grad = new double[d];

      if (Main.fully_supervised) {
//        double[] gradient = the_model_for_each_x.phi(z, x); // TODO: recover this
          double[] gradient = new double[d];
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

      if(Main.gradient_descent_type == Main.DECREASING_STEP_SIZES) {
        eta = Main.eta0 * 1.00/Math.sqrt(counter);
      } else if (Main.gradient_descent_type == Main.CONSTANT_STEP_SIZES){
        eta = Main.eta0;
      }

      for(int i = 0; i < d; i++) {
        theta_hat[i] += eta * new_grad[i];
        theta_hat_average[i] =
            (counter - 1)/(double)counter*theta_hat_average[i] +
            1/(double)counter*theta_hat[i];
      }
      
      if((Main.log_likelihood_verbose && counter % 100 == 0) || (counter == train_data.size())) {
        double average_log_likelihood = calculate_average_log_likelihood(train_data, theta_hat_average);
        System.out.println(
            counter + 
            ": train dataset average log-likelihood:\t" +
            average_log_likelihood
        );
      }
      if(Main.learning_verbose) {
        System.out.println(counter + ": theta_hat_average:\t" + Arrays.toString(theta_hat_average));
      }

    }
    LogInfo.end_track("train");
    return theta_hat_average;
  }
  
  public double calculate_average_log_likelihood(ArrayList<Example> train_data, double theta[]) {
    double total_log_likelihood = 0.0;
    int count = 0;
    for (Example sample : train_data) {
      count += 1;
      int[] y = (int[]) sample.getOutput();
      int[] z = (int[]) sample.getOutput(); // different semantics
      int[] x = (int[]) sample.getInput();
      if(Main.fully_supervised) {
        total_log_likelihood += Math.log(p(z, x, theta));
      } else {
        total_log_likelihood += Math.log(real_p(y, x, theta, Main.xi));
      }
    }
    return total_log_likelihood/(double)count;
  }
  
  public double real_p(int[] y, int[] x, double[] theta, double xi) {
    // TODO
    return 0.5;
  }
    
  public double p(int[] z, int[] x, double[] theta) {
    // TODO
    return 1;
  }
  
  public double[] expectation_phi(int[] x, double[] theta) {
    double[] total = new double[d];
//    int[] indicesForZ = new int[numY];
//    for(int i = 0; i < numY; i++) {
//      indicesForZ[i] = numY; //TODO: check this
//    }
//    if(approx_inference == Main.EXACT) {
//      for(int[] z : new CartesianProduct(indicesForZ)) {
//        double[] the_phi = phi(z, x);
//        for (int k = 0; k < d; k++)
//          total[k] += p(z, x, theta) * the_phi[k];
//      }
//    }
    return total;
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
}
