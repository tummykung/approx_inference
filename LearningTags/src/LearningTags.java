/**
* Learn POS tags using the forward-backward algorithm and a linear-chain CRF model
* @author Tum Chaturapruek
*/
import java.io.*;
import java.util.*;
import java.lang.*;

import fig.basic.*;
import fig.exec.*;
import fig.prob.SampleUtils;
import util.CounterMap;
import util.Counter;
import util.Triplet;

public class LearningTags {

  private static final String TRAIN_FILEPATH = "data/231/tiny";
  private static final String TEST_FILEPATH = "data/231/test";
  private static final boolean DEBUG = false;
  private CounterMap<String, String> wordAndPOSFrequencies = new CounterMap<String, String>();
  private Counter<String> POSFrequencies = new Counter<String>();
  private Counter<String> wordsFrequencies = new Counter<String>();

  private CounterMap<String, String> wordAndPOSWFrequencies = new CounterMap<String, String>();
  private Counter<String> POSWFrequencies = new Counter<String>();
  private int D;
  private int dimPOS;   // in training
  private int dimWords; // in training
  private Map<String, Integer> wordIndex;
  private Map<String, Integer> POSIndex;
  private ArrayList<String> words;
  private ArrayList<String> POSes;
  private double[] finalTheta;
  private ArrayList<Sentence> trainSentences = new ArrayList<Sentence>();
  private ArrayList<Sentence> testSentences = new ArrayList<Sentence>();
  private static String predictor = "CRF"; // Either unigram or CRF


  public static void main(String[] args) throws Exception {
    LearningTags learningTags = new LearningTags();

    System.out.println("dimPOS: " + learningTags.dimPOS);
    System.out.println("dimWords: " + learningTags.dimWords);
    System.out.println("D = " + learningTags.dimPOS * learningTags.dimPOS +
        learningTags.dimPOS * learningTags.dimWords);
    System.out.println("numSentences = " + learningTags.trainSentences.size());
    
    if (predictor.equals("CRF")) {
      learningTags.train(args);
    }
    learningTags.evaluate();
  }

  public LearningTags() {
    readInput(TRAIN_FILEPATH, trainSentences);
    readInput(TEST_FILEPATH, testSentences);
    calculateUnigrams();
    setup();
  }

  public void readInput(String filename, ArrayList<Sentence> outputVariable) {
    BufferedReader br;
    String currentLine;

    try {
      br = new BufferedReader(new FileReader(filename));
      while ((currentLine = br.readLine()) != null) {
        String[] the_words = currentLine.split(" ");
        Sentence sentence = new Sentence();
        while(the_words.length > 1) {
          // then we know it's 3.
          Token currentToken = new Token();
          currentToken.x = the_words[0];
          currentToken.y = the_words[1];
          currentToken.z = the_words[2];

          sentence.addToken(currentToken);

          currentLine = br.readLine();
          the_words = currentLine.split(" ");
        }
        outputVariable.add(sentence);
      }
      br.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void calculateUnigrams() {
    for (Sentence sentence : trainSentences) {
      ArrayList<String> Xs = sentence.getAllX();
      ArrayList<String> Ys = sentence.getAllY();
      ArrayList<String> Zs = sentence.getAllZ();

      for (int i = 0; i < Xs.size(); i++) {
        wordAndPOSFrequencies.incrementCount(Xs.get(i), Ys.get(i), 1);
        wordAndPOSWFrequencies.incrementCount(Xs.get(i), Zs.get(i), 1);
        wordsFrequencies.incrementCount(Xs.get(i), 1);
        POSFrequencies.incrementCount(Ys.get(i), 1);
        POSWFrequencies.incrementCount(Zs.get(i), 1);
      }
    }
  }

  public void setup() {
    dimPOS = POSFrequencies.size();
    dimWords = wordsFrequencies.size();
    D = dimPOS * dimPOS + dimPOS * dimWords;

    Set<String> POSTemp = POSFrequencies.keySet();
    Set<String> wordsTemp = wordsFrequencies.keySet();

    wordIndex = new HashMap<String, Integer>();
    POSIndex = new HashMap<String, Integer>();
    words = new ArrayList<String>();
    POSes = new ArrayList<String>();
    int wordCounter = 0;
    for(String word : wordsTemp) {
      words.add(word);
      wordIndex.put(word, wordCounter);
      wordCounter += 1;
    }

    int POSCounter = 0;
    for(String POS : POSTemp) {
      POSes.add(POS);
      POSIndex.put(POS, POSCounter);
      POSCounter += 1;
    }

    finalTheta = new double[D];
  }

  public int indexTransition(int i, int j) {
    assert(0 <= i && i < dimPOS);
    assert(0 <= j && j < dimPOS);
    return dimPOS*i + j;
  }

  public int indexEmission(int i, int j) {
    assert(0 <= i && i < dimPOS);
    assert(0 <= j && j < dimWords);
    return dimPOS*dimPOS + dimWords*i + j;
  }

  // public Pair<Integer, Integer> inverseIndexTransition(int index) {
  //   int i = index/dimPOS;
  //   int j = index - dimPOS*i;
  //   return new Pair<Integer, Integer>(i, j);
  // }

  // public Pair<Integer, Integer> inverseIndexEmission(int index) {
  //   index -= dimPOS*dimPOS;
  //   int i = index/dimWords;
  //   int j = index - dimWords*i;
  //   return new Pair<Integer, Integer>(i, j);
  // }

  public void train(String[] args) {
    final String y0 = "NN";
    final ArrayList<String> y0Set = new ArrayList<String>();
    y0Set.add(y0);

    BacktrackingLineSearch.Options btopts = new BacktrackingLineSearch.Options();
    LBFGSMaximizer.Options lopts = new LBFGSMaximizer.Options();
    // Execution.init(args, "lsearch", btopts, "lbfgs", lopts);

    //Maximizer maximizer = new GradientMaximizer(btopts);
    Maximizer maximizer = new LBFGSMaximizer(btopts, lopts);
    System.out.println();

    Maximizer.FunctionState state = new Maximizer.FunctionState() {
      // final int D = dimPOS * dimPOS + dimPOS * dimWords; // Dimensionality of the problem

      // Current theta, value, gradient
      private double[] theta = new double[D];
      private double value;
      private double[] gradient = new double[D];
      boolean valueValid = false;
      boolean gradientValid = false;
      boolean cacheValid = false;
      // Map<Triplet<Integer, Integer, String>, Double> logAlphas;
      // Map<Triplet<Integer, Integer, String>, Double> logBetas;
      Double[][] logAlphaArray;
      Double[][] logBetaArray;
      // Double[] logZs;
      double logZ;
      private int sentenceCounter = 0;

      public void updateSentenceCounter() {
        System.out.println(sentenceCounter);
        sentenceCounter = sentenceCounter + 1;
        if (sentenceCounter == trainSentences.size()) {
          sentenceCounter = 0;
        }
      }

      public double[] point() { return theta; }

      public void calculateCachesSentence(Sentence sentence) {
        // start these arrays fresh!
          logAlphaArray = new Double[sentence.length() + 1][dimPOS];
          logBetaArray = new Double[sentence.length() + 1][dimPOS];
          for(int k = 0; k < 10; k ++) {
            System.out.println("=============================================");
          }
          System.out.println("sentence.length() + 1 = " + sentence.length() + 1);
          System.out.println("logBetaArray.length = " + logBetaArray.length);
          System.out.println("logBetaArray[0].length = " + logBetaArray[0].length);
          System.out.println("=============================================");

          ArrayList<String> xs = new ArrayList<String>();
          xs.add(""); // it doesn't really matter what x0 is. We just want to align xs with ys.
          xs.addAll(sentence.getAllX());
          ArrayList<String> ys = new ArrayList<String>();
          ys.add(y0);
          ys.addAll(sentence.getAllY());

          // ---------------------- computing logAlphas -------------------
          // base case
          for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
            // part1 = theta_e(b, xs.get(1))
            double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(1)))];
            // part2 = theta_t(y0, b)
            double part2 = theta[indexTransition(POSIndex.get(y0), bIndex)];
            logAlphaArray[1][bIndex] = part1 + part2;
            // System.out.println("k = " + 1);
            // System.out.println("bIndex = " + bIndex);
            // logAlphas.put(new Triplet<Integer, Integer, String>(sentenceCounter, 1, b), part1 + part2);
          }
          // System.out.println("-------");
          // System.out.println(logAlphaArray[1][0]);
          // System.out.println("-------");
          // recurrence
          for (int k = 1; k <= sentence.length() - 1; k++) {
            for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
              ArrayList<Double> exponents = new ArrayList<Double>();
              for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
                // part1 = theta_e(b, xs.get(k + 1))
                double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(k + 1)))];
                // part2 = theta_t(a, b)
                double part2 = theta[indexTransition(aIndex, bIndex)];
                // System.out.println("k = " + k);
                // System.out.println("aIndex = " + aIndex);
                double part3 = logAlphaArray[k][aIndex];
                // double part3 = logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, k, a));
                exponents.add(part1 + part2 + part3);
              }

              // use the log sum exp trick to stabilize numerics:
              // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
              double maxExponent = Double.NEGATIVE_INFINITY;
              for(double exponent : exponents) {
                if(exponent > maxExponent) {
                  maxExponent = exponent;
                }
              }
              double newValue = 0;
              for(double exponent : exponents) {
                newValue += Math.exp(exponent - maxExponent);
              }
              double newLogAlpha = maxExponent + Math.log(newValue);
              if (DEBUG) {
                System.out.println("newLogAlpha = " + newLogAlpha);
              }
              logAlphaArray[k + 1][bIndex] = newLogAlpha;
              // logAlphas.put(new Triplet<Integer, Integer, String>(sentenceCounter, k + 1, b), newLogAlpha);
            }
          }

          // ---------------------- computing logBetas -------------------
          // base case
          for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
            logBetaArray[sentence.length()][aIndex] = 0.00;
            // logBetas.put(new Triplet<Integer, Integer, String>(sentenceCounter, sentence.length(), a), 0.00);
          }

          // recurrence
          for (int k = sentence.length(); k >= 1; k--) {
            double newValue = 0;

            if (k > 1) {
              for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
                ArrayList<Double> exponents = new ArrayList<Double>();
                for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
                  // part1 = theta_e(a, xs.get(k))
                  double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(k)))];
                  // part2 = theta_t(a, b)
                  double part2 = theta[indexTransition(aIndex, bIndex)];
                  double part3 = logBetaArray[k][bIndex];
                  // double part3 = logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, k, b));
                  exponents.add(part1 + part2 + part3);
                }

                // use the log sum exp trick to stabilize numerics:
                // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
                double maxExponent = Double.NEGATIVE_INFINITY;
                for(double exponent : exponents) {
                  if(exponent > maxExponent) {
                    maxExponent = exponent;
                  }
                }
                double newValue1 = 0;
                for(double exponent : exponents) {
                  newValue1 += Math.exp(exponent - maxExponent);
                }
                double newLogBeta = maxExponent + Math.log(newValue1);
                if (DEBUG) {
                  System.out.println("newLogBeta = " + newLogBeta);
                }
                logBetaArray[k - 1][aIndex] = newLogBeta;
                // logBetas.put(new Triplet<Integer, Integer, String>(sentenceCounter, k - 1, a), newLogBeta);
              }
            } else {
              ArrayList<Double> exponents = new ArrayList<Double>();
              for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
                // part1 = theta_e(a, xs.get(k))
                double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(k)))];
                // part2 = theta_t(a, y0)
                double part2 = theta[indexTransition(POSIndex.get(y0), bIndex)];
                double part3 = logBetaArray[k][bIndex];
                // double part3 = logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, k, b));
                exponents.add(part1 + part2 + part3);
              }

              // use the log sum exp trick to stabilize numerics:
              // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
              double maxExponent = Double.NEGATIVE_INFINITY;
              for(double exponent : exponents) {
                if(exponent > maxExponent) {
                  maxExponent = exponent;
                }
              }
              double newValue1 = 0;
              for(double exponent : exponents) {
                newValue1 += Math.exp(exponent - maxExponent);
              }
              double newLogBeta = maxExponent + Math.log(newValue1);
              logBetaArray[k - 1][POSIndex.get(y0)] = newLogBeta;
              // logBetas.put(new Triplet<Integer, Integer, String>(sentenceCounter, k - 1, y0), newLogBeta);
            }
          }

          logZ = logBetaArray[0][POSIndex.get(y0)];
          // double logZ = logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, 0, y0));
          
          if (DEBUG) {
            // we'll calculate it another way to double check
            ArrayList<Double> exponents = new ArrayList<Double>();
            for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
              exponents.add(logAlphaArray[sentence.length()][aIndex]);
              // exponents.add(logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, sentence.length(), a)));
            }
            // use the log sum exp trick to stabilize numerics:
            // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
            double maxExponent = Double.NEGATIVE_INFINITY;
            for(double exponent : exponents) {
              if(exponent > maxExponent) {
                maxExponent = exponent;
              }
            }
            double newValue = 0;
            for(double exponent : exponents) {
              newValue += Math.exp(exponent - maxExponent);
            }
            double alternativeLogZ = maxExponent + Math.log(newValue);
            double tolerance = Math.pow(10, -7);
            System.out.println("logZ = " + logZ);
            assert(Math.abs(logZ - alternativeLogZ) < tolerance);
          }
      }
      public void calculateCaches() {
        if(!cacheValid) {
          int sentenceCounter = 0;
          calculateCachesSentence(trainSentences.get(sentenceCounter));
          cacheValid = true;
        }
      }

      public double[] gradientSentence(Sentence sentence) {
        ArrayList<String> xs = new ArrayList<String>();
            xs.add("");
            xs.addAll(sentence.getAllX());
            ArrayList<String> ys = new ArrayList<String>();
            ys.add(y0);
            ys.addAll(sentence.getAllY());

            if (DEBUG) {
              // check for correctness: sum_a p(y_i = a | x) = 1.
              for(int i = 1; i <= sentence.length(); i++) {
                double theValue = 0;
                for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
                  // theValue += Math.exp(
                  //   logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, a))
                  //     +
                  //   logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, a))
                  //   -
                  //   logZs.get(sentenceCounter)
                  // );
                  theValue += Math.exp(
                    logAlphaArray[i][aIndex] +
                    logBetaArray[i][aIndex] -
                    logZ
                  );
                }
                double tolerance = Math.pow(10, -7);
                assert(Math.abs(theValue - 1.00) < tolerance);
              }
            }

            for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
              for(int bIndex = 0; bIndex < dimPOS; bIndex++) {
                double newValue = 0;
                for(int i = 1; i <= sentence.length(); i++) {
                    // add \I[y_{i-1}^{(n)}=a,y_{i}^{(n)}=b]
                  if(ys.get(i - 1).equals(POSes.get(aIndex)) && ys.get(i).equals(POSes.get(bIndex))) {
                    newValue += 1;
                  }

                  if(i > 1) {
                    System.out.println("sentence.length() + 1 = " + sentence.length() + 1);
                    System.out.println("dimPOS = " + dimPOS);
                    System.out.println("i = " + i);
                    System.out.println("aIndex = " + aIndex);
                    System.out.println("bIndex = " + bIndex);
                    System.out.println("logAlphaArray[i - 1][aIndex] = " + logAlphaArray[i - 1][aIndex]);
                    System.out.println("theta[indexEmission(bIndex, wordIndex.get(xs.get(i)))] = " + theta[indexEmission(bIndex, wordIndex.get(xs.get(i)))]);
                    System.out.println("theta[indexTransition(aIndex, bIndex)] = " + theta[indexTransition(aIndex, bIndex)]);
                    System.out.println("logBetaArray.length = " + logBetaArray.length);
                    System.out.println("logBetaArray[i][bIndex] = " + logBetaArray[i][bIndex]);
                    newValue -= Math.exp(
                      logAlphaArray[i - 1][aIndex] +
                      theta[indexEmission(bIndex, wordIndex.get(xs.get(i)))] +
                      theta[indexTransition(aIndex, bIndex)] +
                      logBetaArray[i][bIndex] -
                      logZ
                    );
                    // newValue -=
                    // Math.exp(
                    //   logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i - 1, a))
                    //     +
                    //   theta[indexEmission(bIndex, wordIndex.get(xs.get(i)))]
                    //     +
                    //   theta[indexTransition(aIndex, bIndex)]
                    //     +
                    //   logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, b))
                    //     -
                    //   logZs.get(sentenceCounter)
                    // );
                  }
                }
                gradient[indexTransition(aIndex, bIndex)] += newValue;
              }
            }

            for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
              for(String c : words) {
                double newValue = 0;
                for(int i = 1; i <= sentence.length(); i++) {
                    // add \I[y_{i-1}^{(n)}=a,x_{i}^{(n)}=c]
                  if(ys.get(i - 1).equals(POSIndex.get(aIndex)) && xs.get(i).equals(c)) {
                    newValue += 1;
                  }

                    // subtract \I[x_{i}^{(n)}=c]p_{\theta}(y_{i}=a\mid\xx^{(n)})

                  if(xs.get(i).equals(c)) {
                    newValue -= Math.exp(
                      logAlphaArray[i][aIndex] +
                      logBetaArray[i][aIndex] -
                      logZ
                    );
                    // newValue -=
                    // Math.exp(
                    //   logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, a))
                    //     +
                    //   logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, a))
                    //   -
                    //   logZs.get(sentenceCounter)
                    // );
                  }
                  gradient[indexEmission(aIndex, wordIndex.get(c))] += newValue;
                }
              }
            }
          return gradient;
      }

      public double[] gradient() {
        if (!gradientValid) {
          calculateCaches();
          gradient = gradientSentence(trainSentences.get(sentenceCounter));
          updateSentenceCounter();
        }
        gradientValid = true;
        return gradient;
      }

      public double value() {
        if (!valueValid) {
          calculateCaches();
          value = 0;
          Sentence sentence = trainSentences.get(sentenceCounter);
          ArrayList<String> xs = new ArrayList<String>();
          xs.add("");
          xs.addAll(sentence.getAllX());
          ArrayList<String> ys = new ArrayList<String>();
          ys.add(y0);
          ys.addAll(sentence.getAllY());

          for(int i = 1; i <= sentence.length(); i++) {
            for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
              for(int bIndex = 0; bIndex < dimPOS; bIndex++) {

                if(i > 1 && ys.get(i - 1).equals(POSes.get(aIndex)) && ys.get(i).equals(POSes.get(bIndex))) {
                  value += theta[indexTransition(aIndex, bIndex)];
                }
              }

              for(String c : words) {
                if(ys.get(i).equals(POSes.get(aIndex)) && xs.get(i).equals(c)) {
                  value += theta[indexEmission(aIndex, wordIndex.get(c))];
                }
              }
            }
          }

          value -= logZ;
        }
        return value;
      }

      public void invalidate() {
        valueValid = gradientValid = cacheValid = false;
      }
    };

    // Optimize!
    for(int iter = 0; ; iter++) {
      for (int i = 0; i < trainSentences.size(); i++) {
        // state.sentenceCounter = i;
        LogInfo.logs("iter %d: theta = %s, value = %s, gradient = %s, trainSentenceCounter = %s", iter,
          Fmt.D(state.point()),
          Fmt.D(state.value()),
          Fmt.D(state.gradient()),
          i
        );
      }
      if(maximizer.takeStep(state)) break;
    }
    // Execution.finish();

    double[] thetaTemp = state.point();
    for(int i = 0; i < D; i++)
    {
      finalTheta[i] = thetaTemp[i];
    }
  }

  public ArrayList<String> predictPOSLabelsUsingUnigram(ArrayList<String> inputs) {
    ArrayList<String> output = new ArrayList<String>();
    String mostFrequentPOSTag = POSFrequencies.argMax();
    for(String word : inputs) {

      Counter<String> POSFrequencies = wordAndPOSFrequencies.getCounter(word);
      String candidate = POSFrequencies.argMax();
      if(candidate != null) {
        output.add(candidate);
      } else {
        output.add(mostFrequentPOSTag);
      }
    }
    return output;
  }

  public ArrayList<String> predictPOSWLabelsUsingUnigram(ArrayList<String> inputs) {
    ArrayList<String> output = new ArrayList<String>();
    String mostFrequentPOSWTag = POSWFrequencies.argMax();
    for(String word : inputs) {

      Counter<String> POSWFrequencies = wordAndPOSWFrequencies.getCounter(word);
      String candidate = POSWFrequencies.argMax();
      if(candidate != null) {
        output.add(candidate);
      } else {
        output.add(mostFrequentPOSWTag);
      }
    }
    return output;
  }

  public ArrayList<String> predictPOSLabels(ArrayList<String> inputs) {
    final int L = inputs.size();
    final String y0 = "NN";

    // Map<Pair<Integer, String>, Double> logDelta = new HashMap<Pair<Integer, String>, Double>();
    Double[][] logDeltaArray = new Double[L + 1][dimPOS];
    ArrayList<String> xs = new ArrayList<String>();
    xs.add("");
    xs.addAll(inputs);
    // base case
    for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
      // part1 = theta_e(b, xs.get(1))
      double part1;
      if (wordIndex.containsKey(xs.get(1))) {
        part1 = finalTheta[indexEmission(bIndex, wordIndex.get(xs.get(1)))];
      } else {
        part1 = 0;
      }
      // part2 = theta_t(y0, b)
      double part2 = finalTheta[indexTransition(POSIndex.get(y0), bIndex)];
      logDeltaArray[1][bIndex] = part1 + part2;
      // logDelta.put(new Pair<Integer, String>(1, b), part1 + part2);
    }
    // recurrence
    for (int k = 1; k < L; k++) {
      double newValue = 0;
      for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
        for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
          // part1 = theta_e(b, xs.get(k + 1))
          double part1;
          if (wordIndex.containsKey(xs.get(k + 1))) {
            part1 = finalTheta[indexEmission(bIndex, wordIndex.get(xs.get(k + 1)))];
          } else {
            part1 = 0;
          }
          // part2 = theta_t(a, b)
          double part2 = finalTheta[indexTransition(aIndex, bIndex)];
          double candidate = logDeltaArray[k][aIndex] + part1 + part2;
          // double candidate = logDelta.get(new Pair<Integer, String>(k, a)) + part1 + part2;
          if(candidate > newValue) {
            newValue = candidate;
          }
        }
        if (false) { // you can turn on once to see the values of logAlphas, but they're too noisy to print all the time
          System.out.println("logDelta = " + newValue);
        }
        logDeltaArray[k + 1][bIndex] = newValue;
        // logDelta.put(new Pair<Integer, String>(k + 1, b), newValue);
      }
    }
    // backward: Viterbi
    ArrayList<String> ys = new ArrayList<String>();
    // we will add from the back and *reverse* the ys array at the end
    double maxValue = Double.NEGATIVE_INFINITY;
    String argmax = "";
    for(int bIndex = 0; bIndex < dimPOS; bIndex++) {
      double candidate = logDeltaArray[L][bIndex];
      // double candidate = logDelta.get(new Pair<Integer, String>(L, b));
      if(candidate > maxValue) {
        maxValue = candidate;
        argmax = POSes.get(bIndex);
      }
    }
    ys.add(argmax);
    for(int i = L - 1; i >= 1; i--) {
      String yIPlusOneStar = ys.get(ys.size() - 1);
      maxValue = Double.NEGATIVE_INFINITY;
      argmax = "";
      for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
        double candidate =
            logDeltaArray[i][aIndex]
          // logDelta.get(new Pair<Integer, String>(i, a))
            +
          finalTheta[indexTransition(aIndex, POSIndex.get(yIPlusOneStar))];
        if(candidate > maxValue) {
          maxValue = candidate;
          argmax = POSes.get(aIndex);
        }
      }
      ys.add(argmax);
    }
    assert(ys.size() == inputs.size());
    Collections.reverse(ys);
    return ys;
  }

  public ArrayList<String> predictPOSWLabels(ArrayList<String> inputs) {
    return inputs;
  }

  public void evaluate() throws Exception {
    ArrayList<Pair<Integer, Integer>> statistics = new ArrayList<Pair<Integer, Integer>>();
    ArrayList<Pair<Integer, Integer>> statisticsW = new ArrayList<Pair<Integer, Integer>>();
    for(Sentence sentence : testSentences) {
      ArrayList<String> words = sentence.getAllX();
      ArrayList<String> predictions;
      if (predictor.equals("unigram")) {
    	  predictions = predictPOSLabelsUsingUnigram(words);
      } else if (predictor.equals("CRF")) {
    	  predictions = predictPOSLabels(words);
      } else {
    	  throw new Exception("Wrong predictor option.");
      }
      ArrayList<String> predictionsW = predictPOSWLabelsUsingUnigram(words);
      ArrayList<String> answers = sentence.getAllY();
      ArrayList<String> answersW = sentence.getAllZ();
      if (DEBUG) {
        System.out.println("****************");
        System.out.println(sentence);
        System.out.println();
        System.out.print("predictions:\t");
        for (int i = 0; i < predictions.size(); i++) {
          if(predictions.get(i).equals(answers.get(i))) {
            System.out.print("(" + words.get(i) + ", " + predictions.get(i) + "), ");
          } else {
            System.out.print("(*" + words.get(i) + ", " + predictions.get(i) + "*), ");
          }
        }
        System.out.println();
        System.out.print("answers:\t");
        for (int i = 0; i < answers.size(); i++) {
          System.out.print("(" + words.get(i) + ", " + answers.get(i) + "), ");
        }
        System.out.println();
      }

      assert(predictions.size() == answers.size());
      int numCorrect = 0;
      int numIncorrect = 0;
      for (int i = 0; i < predictions.size(); i++) {
        String prediction = predictions.get(i);
        String answer = answers.get(i);
        if (prediction.equals(answer)) {
          numCorrect += 1;
        } else {
          numIncorrect += 1;
        }
      }
      if (DEBUG) {
        System.out.printf("%s:\t%.4f%n", "Precision", numCorrect/(double)(numCorrect + numIncorrect));
      }
      statistics.add(new Pair<Integer, Integer>(numCorrect, numIncorrect));

      if (DEBUG) {
        System.out.println("----------------");
        System.out.print("WSJ predictions:\t");
        for (int i = 0; i < predictionsW.size(); i++) {
          if(predictionsW.get(i).equals(answersW.get(i))) {
            System.out.print("(" + words.get(i) + ", " + predictionsW.get(i) + "), ");
          } else {
            System.out.print("(*" + words.get(i) + ", " + predictionsW.get(i) + "*), ");
          }
        }
        System.out.println();
        System.out.print("WSJ answersW:\t");
        for (int i = 0; i < answersW.size(); i++) {
          System.out.print("(" + words.get(i) + ", " + answersW.get(i) + "), ");
        }
        System.out.println();
      }

      assert(predictionsW.size() == answersW.size());
      int numCorrectW = 0;
      int numIncorrectW = 0;
      for (int i = 0; i < predictionsW.size(); i++) {
        String predictionW = predictionsW.get(i);
        String answerW = answersW.get(i);
        if (predictionW.equals(answerW)) {
          numCorrectW += 1;
        } else {
          numIncorrectW += 1;
        }
      }
      if (DEBUG) {
        System.out.printf("%s:\t%.4f%n", "Precision", numCorrect/(double)(numCorrectW + numIncorrectW));
      }
      statisticsW.add(new Pair<Integer, Integer>(numCorrectW, numIncorrectW));
    }

    int totalCorrect = 0;
    int totalIncorrect = 0;
    int totalExactMatches = 0;
    for(Pair<Integer, Integer> statistic : statistics) {

      int numCorrect = statistic.getFirst();
      int numIncorrect = statistic.getSecond();

      totalCorrect += numCorrect;
      totalIncorrect += numIncorrect;

      if (numIncorrect == 0) {
        totalExactMatches += 1;
      }
    }

    int totalCorrectW = 0;
    int totalIncorrectW = 0;
    int totalExactMatchesW = 0;
    for(Pair<Integer, Integer> statisticW : statisticsW) {

      int numCorrectW = statisticW.getFirst();
      int numIncorrectW = statisticW.getSecond();

      totalCorrectW += numCorrectW;
      totalIncorrectW += numIncorrectW;

      if (numIncorrectW == 0) {
        totalExactMatchesW += 1;
      }
    }

    System.out.println();
    System.out.println("### Training Profile ###");
    System.out.println("dimPOS: " + dimPOS);
    System.out.println("dimWords: " + dimWords);
    System.out.println("D = " + dimPOS * dimPOS +
        dimPOS * dimWords);
    System.out.println("numSentences = " + trainSentences.size());
    System.out.println("### Evaluation Results ###");
    System.out.println("--- Brill Tagger ---");
    System.out.println("TotalCorrect:\t\t" + totalCorrect);
    System.out.println("TotalIncorrect:\t\t" + totalIncorrect);
    System.out.println("totalExactMatches:\t" + totalExactMatches + "/" + statistics.size() + " = " +
      totalExactMatches/(double)statistics.size());
    System.out.printf("%s:\t\t%.4f%n", "Precision", totalCorrect/(double)(totalCorrect + totalIncorrect));
    System.out.println("--- WSJ Corpus ---");
    System.out.println("TotalCorrect:\t\t" + totalCorrectW);
    System.out.println("TotalIncorrect:\t\t" + totalIncorrectW);
    System.out.println("totalExactMatches:\t" + totalExactMatchesW + "/" + statisticsW.size() + " = " +
      totalExactMatchesW/(double)statisticsW.size());
    System.out.printf("%s:\t\t%.4f%n", "Precision", totalCorrectW/(double)(totalCorrectW + totalIncorrectW));
  }

  public static class Token {
    public String x = ""; // an input word
    public String y = ""; // POS from the Brill tagger
    public String z = ""; // POS from the WSJ corpus
  }

  public static class Sentence {
    public ArrayList<Token> tokens;
    public ArrayList<String> Xs;
    public ArrayList<String> Ys;
    public ArrayList<String> Zs;

    public Sentence() {
      tokens = new ArrayList<Token>();
      Xs = new ArrayList<String>();
      Ys = new ArrayList<String>();
      Zs = new ArrayList<String>();
    }

    public Sentence(ArrayList<Token> tokens) { this.tokens = tokens; }

    public void addToken(Token token) {
      tokens.add(token);
      Xs.add(token.x);
      Ys.add(token.y);
      Zs.add(token.z);
    }

    public Token getToken(int i) { return tokens.get(i); }

    public ArrayList<Token> getAllTokens(Token token) { return tokens; }

    public int length() { return tokens.size(); }

    public ArrayList<String> getAllX() { return Xs; }

    public ArrayList<String> getAllY() { return Ys; }

    public ArrayList<String> getAllZ() { return Zs; }

    @Override
    public String toString() {
      String returned_string = "";
      for (Token token : tokens) {
        returned_string += token.x + " ";
      }
      return returned_string;
    }
  }
}
