import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import util.Pair;
import edu.stanford.nlp.stats.Counter;
import fig.basic.LogInfo;
import fig.basic.NumUtils;

public class ModelAndLearning {
  private ArrayList<Example> data;
  ForwardBackward the_model_for_each_x;
  private Random randomizer;

  // 0 = constant step size
  // 1 = decreasing step size
 
  private int K; // the number of features
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

  public ArrayList<Example> generate_data() {
    // return[n][0] = x is int[] and return[n][1] = y is also int[]
    // and n is the num sample iterator
    // TODO:  generate x from a uniform distribution and sample y.

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
    ForwardBackward the_model_for_each_x = new ForwardBackward(
        edgeWeights,
        nodeWeights
    );
    the_model_for_each_x.infer();

    for (int i = 0; i < Main.num_samples; i++) {
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
    int featureTemplateNum = Main.sentenceLength;
    Params params = new Params(Main.rangeX, Main.rangeY, featureTemplateNum);
    K = Main.rangeY * Main.rangeY + Main.rangeY * Main.rangeX;
    // returns: learned_theta
    double[] theta_hat = new double[K];
    double[] theta_hat_average = new double[K];
    int counter = 0;
    for (Example sample : train_data) {
      counter += 1;
      int[] y = sample.getOutput();
      int[] z = sample.getOutput(); // different semantics, used for fully_supervised case
      int[] x = sample.getInput();
      int L = x.length;
      double[] new_grad = new double[K];

      if (Main.fully_supervised) {
//        double[] gradient = the_model_for_each_x.phi(z, x); // TODO: recover this
          double[] gradient = new double[K];
//        Z = calculate_Z(x, theta_hat)
//        E_grad_phi = expectation_phi(x, theta_hat, Z, approx_inference) // old version


          double[][] nodeWeights = new double[x.length][Main.rangeX];
          
          // TODO: check that nodeWeights is correct
          for (int i = 0; i < L; i++) {
            for (int a = 0; a < Main.rangeZ; a++) {
              nodeWeights[i][a] = params.emissions[i][a][x[i]];
            }
          }

          ForwardBackward the_model_for_each_x = new ForwardBackward(
              params.transitions,
              nodeWeights
          );
          the_model_for_each_x.infer();

          double[] E_grad_phi = expectation_cap_phi(x, theta_hat, params, the_model_for_each_x);
          double[] grad_log_Z = new double[K];
          for(int k = 0; k < K; k++)
            grad_log_Z[k] = gradient[k] - E_grad_phi[k];
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

      if (Main.gradient_descent_type == Main.DECREASING_STEP_SIZES) {
        eta = Main.eta0 * 1.00/Math.sqrt(counter);
      } else if (Main.gradient_descent_type == Main.CONSTANT_STEP_SIZES){
        eta = Main.eta0;
      }

      for (int i = 0; i < K; i++) {
        theta_hat[i] += eta * new_grad[i];
        theta_hat_average[i] =
            (counter - 1)/(double)counter*theta_hat_average[i] +
            1/(double)counter*theta_hat[i];
      }
      if ((Main.log_likelihood_verbose && counter % 100 == 0) || (counter == train_data.size())) {
        double average_log_likelihood = calculate_average_log_likelihood(train_data, theta_hat_average);
        LogInfo.logs(
            counter + 
            ": train dataset average log-likelihood:\t" +
            average_log_likelihood
        );
      }
      if (Main.learning_verbose) {
        LogInfo.logs(counter + ": theta_hat_average:\t" + Arrays.toString(theta_hat_average));
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
    return 0.9;
  }
    
  public double p(int[] z, int[] x, double[] theta) {
    // TODO
    return 1;
  }

  public double[] expectation_cap_phi(int[] x, double[] theta, Params params, ForwardBackward fwbw) {
    double[] total = new double[K];
    int L = x.length;

    if(Main.inferType == Main.EXACT) {
      for(int k = 0; k < K; k++) {
        double the_sum = 0.0;
        for(int i = 0; i < L - 1; i++) {
          double[][] edgePosteriors = new double[Main.rangeY][Main.rangeY];
          fwbw.getEdgePosteriors(i, edgePosteriors);
          for(int yi = 0; yi < Main.rangeY; yi++) {
            for(int yiminus1 = 0; yiminus1 < Main.rangeY; yiminus1++) {
              System.out.println("params.emissions[i][yi][yiminus1] = " + params.emissions[i][yi][yiminus1]);
              System.out.println("edgePosteriors[yi][yiminus1] = " + edgePosteriors[yi][yiminus1]);
              double part1 = params.emissions[i][yi][yiminus1];
              double part2 = edgePosteriors[yi][yiminus1];
              
              if(Main.sanity_check) {
                NumUtils.assertIsFinite(part1);
                NumUtils.assertIsFinite(part2);
              }
              
              the_sum += part1 * part2;
            }
          }
        }
        for(int i = 0; i < L; i++) {
          double[] nodePosteriors = new double[Main.rangeY];
          fwbw.getNodePosteriors(i, nodePosteriors);
        }
        total[k] = the_sum;
        System.out.println("the_sum = " + the_sum);
      }
      System.out.println("total = " + Arrays.toString(total));
    }
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
