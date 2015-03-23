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
    // do nothing
  }

  public Params trainFullSupervision(ArrayList<FullSupervisionExample> trainData) throws Exception {
    return train(castFromFullSupervisionToExample(trainData));
  }
  
  public ArrayList<Example> castFromFullSupervisionToExample(ArrayList<FullSupervisionExample> lst) {
    ArrayList<Example> ret = new ArrayList<Example>();
    for (FullSupervisionExample example : lst) {
      ret.add(example);
    }
    return ret;
  }
  public ArrayList<Example> castFromIndirectSupervisionToExample(ArrayList<IndirectSupervisionExample> lst) {
    ArrayList<Example> ret = new ArrayList<Example>();
    for (IndirectSupervisionExample example : lst) {
      ret.add(example);
    }
    return ret;
  }
  public ArrayList<Example> castFromAlignmentExampleToExample(ArrayList<AlignmentExample> lst) {
    ArrayList<Example> ret = new ArrayList<Example>();
    for (AlignmentExample example : lst) {
      ret.add(example);
    }
    return ret;
  }

  public Params train(ArrayList<Example> trainData) throws Exception {
    LogInfo.begin_track("train");
    if (trainData.size() == 0) {
      throw new Exception("Need at least one train data sample!");
    }
    Main.sentenceLength = trainData.get(0).getInput().length;
    // for now, assume all sentences are of 
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
      int[] y = sample.getOutput();
      int[] x = sample.getInput();
      int L = x.length;
      Params gradient = new Params(Main.rangeX, Main.rangeY);

      if (Main.fullySupervised) {
        //initialize
        Params positiveGradient = new Params(Main.rangeX, Main.rangeY);
        for(int i = 0; i < L; i++) {
          if (i < L - 1)
            positiveGradient.transitions[y[i]][y[i + 1]] += 1;
          positiveGradient.emissions[y[i]][x[i]] += 1;
        }
        if (Main.extraVerbose) {
          LogInfo.logs("x:\t" + Fmt.D(x));
          LogInfo.logs("y:\t" + Fmt.D(y));
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
        throw new Exception("Then we shouldn't be here.");
      }

      GradientUpdate.update(thetaHat, thetaHatAverage, gradient, st, counter, Main.eta0);

      if ((Main.logLikelihoodVerbose && counter % 100 == 0) || (counter == trainData.size())) {
        double averageLogLikelihood;
        if (Main.usingAveragingMethod) {
          averageLogLikelihood = calculateAverageLogLikelihood(trainData, thetaHatAverage);
        } else {
          averageLogLikelihood = calculateAverageLogLikelihood(trainData, thetaHat);
        }
        LogInfo.logs(
            "trainDatasetAverageLogLikelihood:\t" +
            counter +
            "\t" +
            Fmt.D(averageLogLikelihood)
        );
      }
      if (Main.learningVerbose) {
        LogInfo.logs("thetaHatAverage:\t" + counter);
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
  
  public Params trainIndirectSupervision(ArrayList<AlignmentExample> trainData) throws Exception {
    LogInfo.begin_track("train");
    if (trainData.size() == 0) {
      throw new Exception("Need at least one train data sample!");
    }
    ArrayList<Example> trainDataCasted = castFromAlignmentExampleToExample(trainData);

    double delta = 10e-4;
    // returns: learned_theta
    Params thetaHat = new Params(Main.rangeX, Main.rangeZ);
    Params thetaHatAverage = new Params(Main.rangeX, Main.rangeZ);
    Params st = new Params(Main.rangeX, Main.rangeZ, delta); // for AdaGrad
    int counter = 0;
    int[] permutedOrder = SampleUtils.samplePermutation(Main.randomizer, trainData.size());
    
    for (int exampleCounter = 0; exampleCounter < trainData.size(); exampleCounter++) {
      AlignmentExample sample = trainData.get(permutedOrder[exampleCounter]);
      counter = exampleCounter + 1;
      int[] y = sample.getOutput();
      int[] z = sample.getLatent();
      int[] x = sample.getInput();
      int L = x.length;


      Params gradient = new Params(Main.rangeX, Main.rangeZ);

      if (!Main.fullySupervised) {
        if(Main.inferType == 1) {
          // initialize
          Params positiveGradient = new Params(Main.rangeX, Main.rangeZ);
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
        } else if(Main.inferType == 3) {
          gradient = differenceBetweenExpectations(x, y, thetaHat, Main.numSamplesToComputeGradients);
        }
      } else {
        throw new Exception("Then we shouldn't be here.");
        }

      GradientUpdate.update(thetaHat, thetaHatAverage, gradient, st, counter, Main.eta0);

      if ((Main.logLikelihoodVerbose && counter % 100 == 0) || (counter == trainData.size())) {
        double averageLogLikelihood;
        if (Main.usingAveragingMethod) {
          averageLogLikelihood = calculateAverageLogLikelihood(trainDataCasted, thetaHatAverage);
        } else {
          averageLogLikelihood = calculateAverageLogLikelihood(trainDataCasted, thetaHat);
        }
        LogInfo.logs(
            "trainDatasetAverageLogLikelihood:\t" +
            counter +
            "\t" +
            Fmt.D(averageLogLikelihood)
        );
      }
      if (Main.learningVerbose) {
        LogInfo.logs("thetaHatAverage:\t" + counter);
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
    double[][] nodeWeights = new double[L][Main.rangeZ + 1];
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

//      System.out.println(AlignmentExample.toStringHumanReadable(y, "y:"));
//      System.out.println(AlignmentExample.toStringHumanReadable(z, "z:"));
//      System.out.println(AlignmentExample.toStringHumanReadable(x, "x:"));
//      System.out.println(Fmt.D(y));
//      System.out.println(Fmt.D(z));
//      System.out.println(Fmt.D(x));
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
    int[] most_likely = modelForX.getViterbi();
    if (Main.fullySupervised) {
      return most_likely;
    } else {
      return AlignmentExample.denotation(most_likely);
    }
  }
  public Report testFullSupervision(ArrayList<FullSupervisionExample> testData, Params params) throws Exception {
    // cast
    ArrayList<Example> castedTrainData = new ArrayList<Example>();
    for (FullSupervisionExample example : testData) {
      castedTrainData.add(example);
    }
    return test(castedTrainData, params);
  }
  public Report testIndirectSupervision(ArrayList<AlignmentExample> testData, Params params) throws Exception {
    // cast
    ArrayList<Example> castedTrainData = new ArrayList<Example>();
    for (AlignmentExample example : testData) {
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
      int[] predictedY = predict(y, params);
      
      if(Main.predictionVerbose) {
        LogInfo.logs(AlignmentExample.toStringHumanReadable(x, "x:\t\t", false));
        LogInfo.logs(AlignmentExample.toStringHumanReadable(y, "y:\t\t", false));
        LogInfo.logs(AlignmentExample.toStringHumanReadable(predictedY, "predictedY:\t", false));
        LogInfo.logs("");
      }
      boolean exactMatch = true;
      if (y.length != predictedY.length)
        exactMatch = false;
      int minLenth = Math.min(predictedY.length, y.length);
      for(int i = 0; i < minLenth; i++) {
        if(x[i] == predictedY[i]) {
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

  public double logIndirectP(int[] y, int[] x, Params params, double xi, ForwardBackward fwbw) throws Exception {
    long M = Main.numSamplesToComputeQ; // number of samples to calculate p(y | x) from the formula
    // p(y | x) = E[q(y | z)] where z is drawn iid from p(z | x)
    double the_sum = 0;
    for(int i = 0; i < M; i++) {
      int[] z = fwbw.sample(Main.randomizer);
      double logQ = logQExpFamily(y, z, params, xi);
      the_sum += Math.exp(logQ);
    }
    return Math.log(the_sum/M);
  }
  
  public double logQ(int[] y, int[] z, Params params, double xi) throws Exception {
    // q(y | z) = I[y = [[z]]]
    boolean exact_match = true;
    int[] denotation = AlignmentExample.denotation(z);
    int L = denotation.length;

    if(denotation.length != y.length)
      exact_match = false;

    for(int i = 0; exact_match && i < L; i++) {
      if(denotation[i] != y[i])
        exact_match = false;
    }

    if (exact_match) {
      return 0;
    } else {
      return Double.NEGATIVE_INFINITY;
    }
  }

  public double logQExpFamily(int[] y, int[] z, Params params, double xi) throws Exception {
    double logTheSum = 0.0;
    int[] denotation = AlignmentExample.denotation(z);
    int L = denotation.length;

    double logZ = xi + Math.log(
        1
        + Math.exp((L + 1)* Math.log(Main.alphabetSize) - xi)
        - Math.exp(-xi)
    );

    // exact match
    boolean exact_match = true;
    if(denotation.length != L)
      exact_match = false;

    int search_length = L;
    if(search_length > y.length) {
      search_length = y.length;
    }
    for(int i = 0; exact_match && i < search_length; i++) {
      if(denotation[i] != y[i])
        exact_match = false;
    }
    
    if (exact_match) {
      logTheSum += xi;
    } else {
      logTheSum += 0;
    }
    double toReturn = logTheSum - logZ;
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
    Params totalParams = new Params(Main.rangeX, Main.rangeZ);
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
    return totalParams;
  }

  public Params differenceBetweenExpectations(
      int[] x, int[] y, Params params, int M) throws Exception {
      Params output = new Params(Main.rangeX, Main.rangeZ);
      Params total1 = new Params(Main.rangeX, Main.rangeZ);
      double totalw = 0;
      Params total2 = new Params(Main.rangeX, Main.rangeZ);

      ForwardBackward fwbw = getForwardBackward(x, params);
      fwbw.infer();

      for(int i = 0; i < M; i++) {
        int[] z = fwbw.sample(Main.randomizer);
        double logQ = logQExpFamily(y, z, params, Main.xi);
        double w = Math.exp(logQ);
        Params the_phi = params;
        totalw += w;
//        total1 += w * the_phi;
//        total2 += the_phi;
        for (int a = 0; a < Main.rangeZ; a++)
          for (int b = 0; b < Main.rangeZ; b++) {
            total1.transitions[a][b] =+ w * the_phi.transitions[a][b];
            total2.transitions[a][b] =+ the_phi.transitions[a][b];
          }
        for (int a = 0; a < Main.rangeZ; a++)
          for (int c = 0; c < Main.rangeX; c++) {
            total1.emissions[a][c] += w * the_phi.emissions[a][c];
            total2.emissions[a][c] += the_phi.emissions[a][c];
          }
      }

      for (int a = 0; a < Main.rangeZ; a++)
        for (int b = 0; b < Main.rangeZ; b++)
          output.transitions[a][b] = total1.transitions[a][b]/totalw - total2.transitions[a][b]/M;

      for (int a = 0; a < Main.rangeZ; a++)
        for (int c = 0; c < Main.rangeX; c++)
          output.emissions[a][c] = total1.emissions[a][c]/totalw - total2.emissions[a][c]/M;
      return output;
  }
}
