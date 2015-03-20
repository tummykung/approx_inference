import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import util.Pair;
import util.Triplet;
import fig.basic.Fmt;
import fig.basic.LogInfo;
import fig.prob.SampleUtils;

public class ModelAndLearning {
  private double eta; // stepsize

  public ModelAndLearning() {

  }
  public Params trainFullSupervision(ArrayList<FullSupervisionExample> trainData) throws Exception {
    // cast
    ArrayList<Example> castedTrainData = new ArrayList<Example>();
    for (FullSupervisionExample example : trainData) {
      castedTrainData.add(example);
    }
    return train(castedTrainData);
  }
  
  public Params train(ArrayList<Example> trainData) throws Exception {
    LogInfo.begin_track("train");
    if (trainData.size() == 0) {
      throw new Exception("Need at least one train data sample!");
    }
    Main.sentenceLength = trainData.get(0).getInput().length;
    // for now, assume all sentences are of 
    // the same length. TODO: generalize this.
    double delta = 10e-4;
    // returns: learned_theta
    Params thetaHat = new Params(Main.rangeX, Main.rangeY);
    Params thetaHatAverage = new Params(Main.rangeX, Main.rangeY);
    Params st = new Params(Main.rangeX, Main.rangeY, delta); // for AdaGrad
    int counter = 0;
    int[] permutedOrder = SampleUtils.samplePermutation(Main.randomizer, trainData.size());
    
    for (int exampleCounter = 0; exampleCounter < trainData.size(); exampleCounter++) {
      Example sample = trainData.get(permutedOrder[exampleCounter]);
      counter = exampleCounter + 1;
//      int[] y = sample.getOutput();
      int[] z = sample.getOutput(); // different semantics, used for fully_supervised case
      int[] x = sample.getInput();
      int L = x.length;
      Params gradient = new Params(Main.rangeX, Main.rangeY);

      if (Main.fullySupervised) {
        //initialize
        Params positiveGradient = new Params(Main.rangeX, Main.rangeY);
        for(int i = 0; i < L; i++) {
          if (i < L - 1)
            positiveGradient.transitions[z[i]][z[i + 1]] += 1;
          positiveGradient.emissions[z[i]][x[i]] += 1;
        }
        if (Main.extraVerbose) {
          LogInfo.logs("x:\t" + Fmt.D(x));
          LogInfo.logs("z:\t" + Fmt.D(z));
          positiveGradient.print("positiveGradient");
        }

        ForwardBackward theModelForEachX = getForwardBackward(x, thetaHat);
        theModelForEachX.infer();

        Params expectationCapPhi = expectationCapPhi(x, thetaHat, theModelForEachX);

        if (Main.extraVerbose) {
          expectationCapPhi.print("expectationCapPhi");
        }
        // subtract
        for (int a = 0; a <= Main.rangeZ; a++)
          for (int b = 0; b <= Main.rangeZ; b++)
            gradient.transitions[a][b] = positiveGradient.transitions[a][b] - expectationCapPhi.transitions[a][b];
        for (int a = 0; a <= Main.rangeZ; a++)
          for (int c = 0; c <= Main.rangeX; c++)
            gradient.emissions[a][c] = positiveGradient.emissions[a][c] - expectationCapPhi.emissions[a][c];
      } else {
        // -- Indirect supervision -- 
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

      if (Main.gradientDescentType == Main.DECREASING_STEP_SIZES) {
        eta = Main.eta0/Math.sqrt(counter);
      } else if (Main.gradientDescentType == Main.CONSTANT_STEP_SIZES){
        eta = Main.eta0;
      }
      // if AdaGrad we'll do at the level of update itself

      if (Main.extraVerbose) {
        gradient.print("gradient");
        thetaHat.print("thetaHat");
        thetaHatAverage.print("thetaHatAverage");
      }

      for (int a = 0; a <= Main.rangeZ; a++)
        for (int b = 0; b <= Main.rangeZ; b++) {
          if (Main.gradientDescentType == Main.ADAGRAD){
            st.transitions[a][b] += gradient.transitions[a][b] * gradient.transitions[a][b];
            thetaHat.transitions[a][b] += Main.eta0/Math.sqrt(st.transitions[a][b]) * gradient.transitions[a][b];
          } else
            thetaHat.transitions[a][b] += eta * gradient.transitions[a][b];
          thetaHatAverage.transitions[a][b] =
              (counter - 1)/(double)counter*thetaHatAverage.transitions[a][b] +
              1/(double)counter*thetaHat.transitions[a][b];
        }
      for (int a = 0; a <= Main.rangeZ; a++)
        for (int c = 0; c <= Main.rangeX; c++) {
          if (Main.gradientDescentType == Main.ADAGRAD){
            st.emissions[a][c] += gradient.emissions[a][c] * gradient.emissions[a][c];
            thetaHat.emissions[a][c] += Main.eta0/Math.sqrt(st.emissions[a][c]) * gradient.emissions[a][c];
          } else
            thetaHat.emissions[a][c] += eta * gradient.emissions[a][c];
          thetaHatAverage.emissions[a][c] =
              (counter - 1)/(double)counter*thetaHatAverage.emissions[a][c] +
              1/(double)counter*thetaHat.emissions[a][c];
        }


      if ((Main.logLikelihoodVerbose && counter % 100 == 0) || (counter == trainData.size())) {
        double averageLogLikelihood;
        if (Main.usingAveragingMethod) {
          averageLogLikelihood = calculateAverageLogLikelihood(trainData, thetaHatAverage);
        } else {
          averageLogLikelihood = calculateAverageLogLikelihood(trainData, thetaHat);
        }
        LogInfo.logs(
            counter + 
            ": trainDatasetAverageLogLikelihood:\t" +
            Fmt.D(averageLogLikelihood)
        );
      }
      if (Main.learningVerbose) {
        LogInfo.logs(counter + ": thetaHatAverage");
        thetaHatAverage.print("thetaHatAverage");
      }

    }
    LogInfo.end_track("train");
    if (Main.usingAveragingMethod) {
      return thetaHatAverage;
    } else {
      return thetaHat;
    }
  }
  
  public static ForwardBackward getForwardBackward(int[] x, Params params) {
    int L = x.length;
    double[][] nodeWeights = new double[x.length][Main.rangeZ + 1];
    double[][] edgeWeights = new double[Main.rangeZ + 1][Main.rangeZ + 1];
    for (int a = 0; a <= Main.rangeZ; a++)
      for (int b = 0; b <= Main.rangeZ; b++) {
        edgeWeights[a][b] = Math.exp(params.transitions[a][b]);
      }
    for (int i = 0; i < L; i++)
      for (int a = 0; a <= Main.rangeZ; a++)
        nodeWeights[i][a] = Math.exp(params.emissions[a][x[i]]);
    return new ForwardBackward(edgeWeights, nodeWeights);
  }
  
  public double calculateAverageLogLikelihood(ArrayList<Example> trainData, Params params) throws Exception {
    double totalLogLikelihood = 0.0;
    int count = 0;
    for (Example sample : trainData) {
      count += 1;
      int[] y = sample.getOutput();
      int[] z = sample.getOutput(); // different semantics
      int[] x = sample.getInput();

      ForwardBackward the_model_for_each_x = getForwardBackward(x, params);
      the_model_for_each_x.infer();
      double logZ = the_model_for_each_x.getLogZ();
      
      if(Main.fullySupervised) {
        totalLogLikelihood += logP(z, x, params, logZ);
      } else {
        totalLogLikelihood += logIndirectP(y, x, params, Main.xi, the_model_for_each_x);
      }
    }
    return totalLogLikelihood/(double)count;
  }

  public int[] predict(int[] x, Params params) {
    ForwardBackward modelForX = getForwardBackward(x, params);
    modelForX.infer();
    return modelForX.getViterbi();
  }
  public Report testFullSupervision(ArrayList<FullSupervisionExample> testData, Params params) throws Exception {
 // cast
    ArrayList<Example> castedTrainData = new ArrayList<Example>();
    for (FullSupervisionExample example : testData) {
      castedTrainData.add(example);
    }
    return test(castedTrainData, params);
  }
  public Report test(ArrayList<Example> testData, Params params) throws Exception {
    int totalExactMatch = 0;
    int totalUnaryMatch = 0;
    int totalPossibleUnaryMatch = 0;

    for (Example example : testData) {
      int[] x = example.getInput();
      example.getOutput();
      int[] y = (int[]) example.getOutput();
      int[] predictedZ = predict(y, params);
      boolean exactMatch = true;
      if (x.length != predictedZ.length)
        exactMatch = false;
      int minLenth = Math.min(x.length, y.length);
      for(int i = 0; i < minLenth; i++) {
        if(x[i] == predictedZ[i]) {
          totalUnaryMatch += 1;
        } else
          exactMatch = false;
        totalPossibleUnaryMatch += 1;
      }
      if (exactMatch)
        totalExactMatch += 1;
    }
    return new Report(
        totalExactMatch,
        testData.size(),
        totalUnaryMatch,
        totalPossibleUnaryMatch,
        calculateAverageLogLikelihood(testData, params)
    );
  }

  public double logIndirectP(int[] y, int[] x, Params params, double xi, ForwardBackward fwbw) {
    int M = 1000;
    for(int i = 0; i < M; i++) {
      int[] z = fwbw.sample(Main.randomizer);
    }

    return 0.9;
  }
  
  public double logQ(int[] y, int[] z, Params params, double logZ) {
    double theSum = 0.0;
    double toReturn = theSum - logZ;
    assert toReturn <= 0;
    return toReturn;
  }
    
  public double logP(int[] z, int[] x, Params params, double logZ) {
    double theSum = 0.0;
    for(int i = 0; i < z.length; i++) {
      if(i < z.length - 1) {
        theSum += params.transitions[z[i]][z[i + 1]];
      }
      theSum += params.emissions[z[i]][(int) x[i]];
    }
    double toReturn = theSum - logZ;
    assert toReturn <= 0;
    return toReturn;
  }

  public Params expectationCapPhi(int[] x, Params params, ForwardBackward fwbw) {
    Params totalParams = new Params(Main.rangeX, Main.rangeZ, Main.sentenceLength);
    int L = x.length;

    if(Main.inferType == Main.EXACT) {
      for(int a = 0; a <= Main.rangeZ; a++) {
        for(int b = 0; b <= Main.rangeZ; b++) {
          double theSum = 0.0;
          for(int i = 0; i < L - 1; i++) {
            double[][] edgePosteriors = new double[Main.rangeZ + 1][Main.rangeZ + 1];
            fwbw.getEdgePosteriors(i, edgePosteriors);
            for(int zi = 0; zi <= Main.rangeZ; zi++) {
              for(int ziminus1 = 0; ziminus1 <= Main.rangeZ; ziminus1++) {
                int part1 = Util.indicator(zi == a && ziminus1 == b);
                // for an indicator function it seems like an overkill
                // to loop over rangeZ^2, because E[I[b]] = Prob[b],
                // but we keep this pattern for the generalization purpose.
                double part2 = edgePosteriors[zi][ziminus1];
                theSum += part1 * part2;
              }
            }
          }
          totalParams.transitions[a][b] = theSum;
        }
      }

      for(int a = 0; a <= Main.rangeZ; a++) {
        for(int c = 0; c <= Main.rangeX; c++) {
          double theSum = 0.0;
          double[] nodePosteriors = new double[Main.rangeZ + 1];
          for(int i = 0; i < L - 1; i++) {
            fwbw.getNodePosteriors(i, nodePosteriors);
            for(int zi = 0; zi <= Main.rangeZ; zi++) {
              int part1 = Util.indicator(zi == a && x[i] == c);
              double part2 = nodePosteriors[zi];
              theSum += part1 * part2;
            }
          }
          totalParams.emissions[a][c] = theSum;
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

  public double differenceBetweenExpectations(
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
