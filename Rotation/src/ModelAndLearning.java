import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import util.Pair;
import edu.stanford.nlp.stats.Counter;
import fig.basic.Fmt;
import fig.basic.LogInfo;
import fig.basic.NumUtils;
import fig.prob.SampleUtils;

public class ModelAndLearning {
  private ArrayList<Example> data;
  ForwardBackward the_model_for_each_x;
  private Random randomizer;

  // 0 = constant step size
  // 1 = decreasing step size
 
  private int K; // the number of features
  private double eta; // stepsize

  public ModelAndLearning() {
    randomizer = new Random(Main.seed);
  }

  public ArrayList<Example> generate_data() {
    // return[n][0] = x is int[] and return[n][1] = y is also int[]
    // and n is the num sample iterator
    // TODO:  generate x from a uniform distribution and sample y.

    LogInfo.begin_track("generate_data");
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

    // FIXME: generate this from some distribution. e.g., a uniform distribution
    // the things below are p(y) only, not p(y | x). but it's good enough
    // for testing emission probability updates.
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
    LogInfo.end_track("generate_data");
    return data;
  }

  public Params train(ArrayList<Example> train_data) {
    LogInfo.begin_track("train");
    int featureTemplateNum = Main.sentenceLength;
    K = Main.rangeY * Main.rangeY + Main.rangeY * Main.rangeX;
    double delta = 10e-4;
    // returns: learned_theta
    Params theta_hat = new Params(Main.rangeX, Main.rangeY, featureTemplateNum);
    Params theta_hat_average = new Params(Main.rangeX, Main.rangeY, featureTemplateNum);
    Params st = new Params(Main.rangeX, Main.rangeY, featureTemplateNum, delta); // for AdaGrad
    int counter = 0;
    int[] permuted_order = SampleUtils.samplePermutation(this.randomizer, train_data.size());
    
    for (int exampleCounter = 0; exampleCounter < train_data.size(); exampleCounter++) {
      Example sample = train_data.get(permuted_order[exampleCounter]);
      counter = exampleCounter + 1;
      int[] y = sample.getOutput();
      int[] z = sample.getOutput(); // different semantics, used for fully_supervised case
      int[] x = sample.getInput();
      int L = x.length;
      Params new_grad = new Params(Main.rangeX, Main.rangeY, featureTemplateNum);

      if (Main.fully_supervised) {
        //initialize
        Params gradient = new Params(Main.rangeX, Main.rangeY, featureTemplateNum);
        for(int i = 0; i < L; i++) {
          if (i < L - 1)
            gradient.transitions[z[i]][z[i + 1]] += 1;
          gradient.emissions[i][z[i]][x[i]] += 1;
        }
        if (Main.extra_verbose) {
          LogInfo.logs("x = " + Arrays.toString(x));
          LogInfo.logs("z = " + Arrays.toString(z));
          gradient.print("gradient");
        }

        // TODO: check that edgeWeights is correct
        double[][] nodeWeights = new double[x.length][Main.rangeZ];
        double[][] edgeWeights = new double[Main.rangeZ][Main.rangeZ];
        for (int a = 0; a < Main.rangeZ; a++)
          for (int b = 0; b < Main.rangeZ; b++)
            edgeWeights[a][b] = Math.exp(theta_hat.transitions[a][b]);
        for (int i = 0; i < L; i++)
          for (int a = 0; a < Main.rangeZ; a++)
            nodeWeights[i][a] = Math.exp(theta_hat.emissions[i][a][x[i]]);
        ForwardBackward the_model_for_each_x = new ForwardBackward(edgeWeights, nodeWeights);
        the_model_for_each_x.infer();

        Params E_grad_phi = expectation_cap_phi(x, theta_hat, the_model_for_each_x);

        // subtract
        for (int a = 0; a < Main.rangeZ; a++)
          for (int b = 0; b < Main.rangeZ; b++)
            new_grad.transitions[a][b] = gradient.transitions[a][b] - E_grad_phi.transitions[a][b];
        for (int i = 0; i < L; i++)
          for (int a = 0; a < Main.rangeZ; a++)
            for (int c = 0; c < Main.rangeX; c++)
              new_grad.emissions[i][a][c] = gradient.emissions[i][a][c] - E_grad_phi.emissions[i][a][c];
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
        eta = Main.eta0/Math.sqrt(counter);
      } else if (Main.gradient_descent_type == Main.CONSTANT_STEP_SIZES){
        eta = Main.eta0;
      }
      // if adagrad we'll do at the level of update itself

      if (Main.extra_verbose) {
        System.out.println("=====BEGIN DEBUG ZONE=====");
        new_grad.print("new_grad");
        theta_hat.print("theta_hat");
        theta_hat_average.print("theta_hat_average");
        System.out.println("=====END DEBUG ZONE=====");
      }

      for (int a = 0; a < Main.rangeZ; a++)
        for (int b = 0; b < Main.rangeZ; b++) {
          if (Main.gradient_descent_type == Main.ADAGRAD){
            theta_hat.transitions[a][b] = Main.eta0/Math.sqrt(st.transitions[a][b]) * new_grad.transitions[a][b];
            st.transitions[a][b] += new_grad.transitions[a][b] * new_grad.transitions[a][b];
          } else
            theta_hat.transitions[a][b] = eta * new_grad.transitions[a][b];
          theta_hat_average.transitions[a][b] =
              (counter - 1)/(double)counter*theta_hat_average.transitions[a][b] +
              1/(double)counter*theta_hat.transitions[a][b];
        }
      for (int i = 0; i < L; i++)
        for (int a = 0; a < Main.rangeZ; a++)
          for (int c = 0; c < Main.rangeX; c++) {
            if (Main.gradient_descent_type == Main.ADAGRAD){
              theta_hat.emissions[i][a][c] = Main.eta0/Math.sqrt(st.emissions[i][a][c]) * new_grad.emissions[i][a][c];
              st.emissions[i][a][c] += new_grad.emissions[i][a][c] * new_grad.emissions[i][a][c];
            } else
              theta_hat.emissions[i][a][c] = eta * new_grad.emissions[i][a][c];
            theta_hat_average.emissions[i][a][c] =
                (counter - 1)/(double)counter*theta_hat_average.emissions[i][a][c] +
                1/(double)counter*theta_hat.emissions[i][a][c];
          }


      if ((Main.log_likelihood_verbose && counter % 100 == 0) || (counter == train_data.size())) {
        double average_log_likelihood = calculate_average_log_likelihood(train_data, theta_hat);
        LogInfo.logs(
            counter + 
            ": train dataset average log-likelihood:\t" +
            Fmt.D(average_log_likelihood)
        );
      }
      if (Main.learning_verbose) {
        LogInfo.logs(counter + ": theta_hat_average");
        theta_hat_average.print("theta_hat_average");
      }

    }
    LogInfo.end_track("train");
    return theta_hat_average;
  }
  
  public double calculate_average_log_likelihood(ArrayList<Example> train_data, Params params) {
    double total_log_likelihood = 0.0;
    int count = 0;
    for (Example sample : train_data) {
      count += 1;
      int[] y = (int[]) sample.getOutput();
      int[] z = (int[]) sample.getOutput(); // different semantics
      int[] x = (int[]) sample.getInput();
     
   // TODO: check that edgeWeights is correct
      double[][] nodeWeights = new double[x.length][Main.rangeZ];
      double[][] edgeWeights = new double[Main.rangeZ][Main.rangeZ];
      for (int a = 0; a < Main.rangeZ; a++)
        for (int b = 0; b < Main.rangeZ; b++)
          edgeWeights[a][b] = Math.exp(params.transitions[a][b]);
      for (int i = 0; i < x.length; i++)
        for (int a = 0; a < Main.rangeZ; a++)
          nodeWeights[i][a] = Math.exp(params.emissions[i][a][x[i]]);
      ForwardBackward the_model_for_each_x = new ForwardBackward(edgeWeights, nodeWeights);
      the_model_for_each_x.infer();
      double logZ = the_model_for_each_x.getLogZ();
      
      if(Main.fully_supervised) {
        total_log_likelihood += logP(z, x, params, logZ);
      } else {
        total_log_likelihood += logIndirectP(y, x, params, Main.xi);
      }
    }
    return total_log_likelihood/(double)count;
  }
  
  public double logIndirectP(int[] y, int[] x, Params params, double xi) {
    // TODO
    return 0.9;
  }
    
  public double logP(int[] z, int[] x, Params params, double logZ) {
    double the_sum = 0.0;
    for(int i = 0; i < z.length; i++) {
      if(i < z.length - 1) {
        the_sum += params.transitions[z[i]][z[i + 1]];
      }
      the_sum += params.emissions[i][z[i]][x[i]];
    }
    double to_return = the_sum - logZ;
    assert to_return < 0;
    return to_return;
  }

  public Params expectation_cap_phi(int[] x, Params params, ForwardBackward fwbw) {
    Params totalParams = new Params(Main.rangeX, Main.rangeY, Main.sentenceLength);
    int L = x.length;

    if(Main.inferType == Main.EXACT) {
      for(int a = 0; a < Main.rangeZ; a++) {
        for(int b = 0; b < Main.rangeZ; b++) {
          double the_sum = 0.0;
          for(int i = 0; i < L - 1; i++) {
            double[][] edgePosteriors = new double[Main.rangeY][Main.rangeY];
            fwbw.getEdgePosteriors(i, edgePosteriors);
            for(int zi = 0; zi < Main.rangeY; zi++) {
              for(int ziminus1 = 0; ziminus1 < Main.rangeZ; ziminus1++) {
                int part1 = Util.indicator(zi == a && ziminus1 == b);
                // for an indicator function it seems like an overkill
                // to loop over rangeZ^2, because E[I[b]] = Prob[b],
                // but we keep this pattern for the generalization purpose.
                double part2 = edgePosteriors[zi][ziminus1];
                
                if(Main.sanity_check) {
                  NumUtils.assertIsFinite(part1);
                  NumUtils.assertIsFinite(part2);
                }
                the_sum += part1 * part2;
              }
            }
          }
          totalParams.transitions[a][b] = the_sum;
        }
      }

      for(int a = 0; a < Main.rangeZ; a++) {
        for(int c = 0; c < Main.rangeX; c++) {
          double the_sum = 0.0;
          for(int i = 0; i < L - 1; i++) {
            double[] nodePosteriors = new double[Main.rangeY];
            fwbw.getNodePosteriors(i, nodePosteriors);
            for(int zi = 0; zi < Main.rangeZ; zi++) {
              int part1 = Util.indicator(zi == a && x[i] == c);
              double part2 = nodePosteriors[zi];
              
              if(Main.sanity_check) {
                NumUtils.assertIsFinite(part1);
                NumUtils.assertIsFinite(part2);
              }
              the_sum += part1 * part2;
            }
          }
          totalParams.transitions[a][c] = the_sum;
        }
      }
    }
//    if(approx_inference == Main.EXACT) {
//      for(int[] z : new CartesianProduct(indicesForZ)) {
//        double[] the_phi = phi(z, x);
//        for (int k = 0; k < d; k++)
//          total[k] += p(z, x, theta) * the_phi[k];
//      }
//    }
    return totalParams;
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
