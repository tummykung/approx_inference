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

  private static final String TRAIN_FILEPATH = "data/231/temp";
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

  public static void main(String[] args) {
    LearningTags learningTags = new LearningTags();
    learningTags.train(args);
    learningTags.evaluate();
  }

  public LearningTags() {
    readInput(TRAIN_FILEPATH, trainSentences);
    readInput(TEST_FILEPATH, testSentences);
    calculateBagOfWords();
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
        while(the_words.length != 1) {
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

  public void calculateBagOfWords() {
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
    final ArrayList<Double> logZs = new ArrayList<Double>();
    y0Set.add(y0);

    System.err.println("dimPOS: " + dimPOS);
    System.err.println("dimWords: " + dimWords);
    System.err.println(dimPOS * dimPOS + dimPOS * dimWords);


    BacktrackingLineSearch.Options btopts = new BacktrackingLineSearch.Options();
    LBFGSMaximizer.Options lopts = new LBFGSMaximizer.Options();
    // Execution.init(args, "lsearch", btopts, "lbfgs", lopts);

    //Maximizer maximizer = new GradientMaximizer(btopts);
    Maximizer maximizer = new LBFGSMaximizer(btopts, lopts);
    System.out.println();

    Maximizer.FunctionState state = new Maximizer.FunctionState() {
      // Specifies the function
      // final int D = dimPOS * dimPOS + dimPOS * dimWords; // Dimensionality of the problem


      // Current theta, value, gradient
      private double[] theta = new double[D];
      private double value;
      private double[] gradient = new double[D];
      boolean valueValid = false;
      boolean gradientValid = false;
      boolean cacheValid = false;
      Map<Triplet<Integer, Integer, String>, Double> logAlphas = new HashMap<Triplet<Integer, Integer, String>, Double>();
      Map<Triplet<Integer, Integer, String>, Double> logBetas = new HashMap<Triplet<Integer, Integer, String>, Double>();

      public double[] point() { return theta; }

      public void calculateCaches() {
        if(!cacheValid) {
          int sentenceCounter = 0;
          for(Sentence sentence : trainSentences) {
            ArrayList<String> xs = new ArrayList<String>();
            xs.add("");
            xs.addAll(sentence.getAllX());
            ArrayList<String> ys = new ArrayList<String>();
            ys.add(y0);
            ys.addAll(sentence.getAllY());

            // ---------------------- computing logAlphas -------------------
            // base case
            for (String b : POSes) {
              // part1 = theta_e(b, xs.get(1))
              double part1 = theta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(1)))];
              // part2 = theta_t(y0, b)
              double part2 = theta[indexTransition(POSIndex.get(y0), POSIndex.get(b))];
              logAlphas.put(new Triplet<Integer, Integer, String>(sentenceCounter, 1, b), part1 + part2);
            }
            // recurrence
            for (int k = 1; k <= sentence.length() - 1; k++) {
              for (String b : POSes) {
                ArrayList<Double> exponents = new ArrayList<Double>();
                for (String a : POSes) {
                  // part1 = theta_e(b, xs.get(k + 1))
                  double part1 = theta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(k + 1)))];
                  // part2 = theta_t(a, b)
                  double part2 = theta[indexTransition(POSIndex.get(a), POSIndex.get(b))];
                  double part3 = logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, k, a));
                  exponents.add(part1 + part2 + part3);
                }

                // use the log sum exp trick to stabilize numerics:
                // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
                double maxExponent = 0;
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
                System.out.println("newLogAlpha = " + newLogAlpha);
                logAlphas.put(new Triplet<Integer, Integer, String>(sentenceCounter, k + 1, b), newLogAlpha);
              }
            }

            // ---------------------- computing logBetas -------------------
            // base case
            for (String a : POSes) {
              logBetas.put(new Triplet<Integer, Integer, String>(sentenceCounter, sentence.length(), a), 0.00);
            }

            // recurrence
            for (int k = sentence.length(); k >= 1; k--) {
              double newValue = 0;

              if (k > 1) {
                for (String a : POSes) {
                  ArrayList<Double> exponents = new ArrayList<Double>();
                  for (String b : POSes) {
                    // part1 = theta_e(a, xs.get(k))
                    double part1 = theta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(k)))];
                    // part2 = theta_t(a, b)
                    double part2 = theta[indexTransition(POSIndex.get(a), POSIndex.get(b))];
                    double part3 = logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, k, b));
                    exponents.add(part1 + part2 + part3);
                  }

                  // use the log sum exp trick to stabilize numerics:
                  // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
                  double maxExponent = 0;
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
                  System.out.println("newLogBeta = " + newLogBeta);
                  logBetas.put(new Triplet<Integer, Integer, String>(sentenceCounter, k - 1, a), newLogBeta);
                }
              } else {
                ArrayList<Double> exponents = new ArrayList<Double>();
                for (String b : POSes) {
                  // part1 = theta_e(a, xs.get(k))
                  double part1 = theta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(k)))];
                  // part2 = theta_t(a, y0)
                  double part2 = theta[indexTransition(POSIndex.get(y0), POSIndex.get(b))];
                  double part3 = logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, k, b));
                  exponents.add(part1 + part2 + part3);
                }

                // use the log sum exp trick to stabilize numerics:
                // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
                double maxExponent = 0;
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
                logBetas.put(new Triplet<Integer, Integer, String>(sentenceCounter, k - 1, y0), newLogBeta);
              }
            }

            double logZ = logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, 0, y0));
            System.out.println("logZ = " + logZ);
            logZs.add(logZ);
            sentenceCounter += 1;
          }
          // end of sentence for loop
          cacheValid = true;
        }
      }

      public double[] gradient() {
        if (!gradientValid) {
          calculateCaches();
          int sentenceCounter = 0;
          for(Sentence sentence : trainSentences) {
            ArrayList<String> xs = new ArrayList<String>();
            xs.add("");
            xs.addAll(sentence.getAllX());
            ArrayList<String> ys = new ArrayList<String>();
            ys.add(y0);
            ys.addAll(sentence.getAllY());

            for(String a : POSes) {
              for(String b : POSes) {
                double newValue = 0;
                for(int i = 1; i <= sentence.length(); i++) {
                    // add \I[y_{i-1}^{(n)}=a,y_{i}^{(n)}=b]
                  if(ys.get(i - 1).equals(a) && ys.get(i).equals(b)) {
                    newValue += 1;
                  }

                  // subtract p_{\theta}(y_{i-1}=a,y_{i}=b\mid\xx^{(n)})
                  // System.out.println("i = " + i);
                  // System.out.println("sentenceCounter = " + sentenceCounter);
                  // System.out.println(logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, b)));
                  // System.out.println(sentence.length());
                  // System.out.println("pass");
                  if(i > 1) {
                    newValue -=
                    Math.exp(
                      logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i - 1, a))
                        +
                      theta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(i)))]
                        +
                      theta[indexTransition(POSIndex.get(a), POSIndex.get(b))]
                        +
                      logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, b))
                        -
                      logZs.get(sentenceCounter)
                    );
                  }
                }
                gradient[indexTransition(POSIndex.get(a), POSIndex.get(b))] += newValue;
              }
            }

            for(String a : POSes) {
              for(String c : words) {
                double newValue = 0;
                for(int i = 1; i <= sentence.length(); i++) {
                    // add \I[y_{i-1}^{(n)}=a,x_{i}^{(n)}=c]
                  if(ys.get(i - 1).equals(a) && xs.get(i).equals(c)) {
                    newValue += 1;
                  }

                    // subtract \I[x_{i}^{(n)}=c]p_{\theta}(y_{i}=a\mid\xx^{(n)})

                  if(xs.get(i).equals(c)) {
                    newValue -=
                    Math.exp(
                      logAlphas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, a))
                        +
                      logBetas.get(new Triplet<Integer, Integer, String>(sentenceCounter, i, a))
                      -
                      logZs.get(sentenceCounter)
                    );
                  }
                  gradient[indexEmission(POSIndex.get(a), wordIndex.get(c))] += newValue;
                }
              }
            }
            sentenceCounter += 1;
          }
          // end of sentence for loop
        }
        gradientValid = true;
        return gradient;
      }

      public double value() {
        if (!valueValid) {
          calculateCaches();
          value = 0;
          int sentenceCounter = 0;
          for(Sentence sentence : trainSentences) {
            ArrayList<String> xs = new ArrayList<String>();
            xs.add("");
            xs.addAll(sentence.getAllX());
            ArrayList<String> ys = new ArrayList<String>();
            ys.add(y0);
            ys.addAll(sentence.getAllY());

            for(int i = 1; i <= sentence.length(); i++) {
              for(String a : POSes) {
                for(String b : POSes) {

                  if(i > 1 && ys.get(i - 1).equals(a) && ys.get(i).equals(b)) {
                    value += theta[indexTransition(POSIndex.get(a), POSIndex.get(b))];
                  }
                }

                for(String c : words) {
                  if(ys.get(i).equals(a) && xs.get(i).equals(c)) {
                    value += theta[indexEmission(POSIndex.get(a), wordIndex.get(c))];
                  }
                }
              }
            }

            value -= logZs.get(sentenceCounter);
            sentenceCounter += 1;
          }
          // end of sentence loop
        }
        return value;
      }

      public void invalidate() { valueValid = gradientValid = cacheValid = false; }
    };

    // Optimize!
    for(int iter = 0; ; iter++) {
      LogInfo.logs("iter %d: theta = %s, value = %s, gradient = %s", iter,
        Fmt.D(state.point()),
        Fmt.D(state.value()),
        Fmt.D(state.gradient()));
      if(maximizer.takeStep(state)) break;
    }
    // Execution.finish();

    double[] thetaTemp = state.point();
    for(int i = 0; i < D; i++)
    {
      finalTheta[i] = thetaTemp[i];
    }
    System.out.println("^^^^^^^^^");
    for(int i = 0; i < D; i++) {
      System.out.print(finalTheta[i] + " ");
    }
    System.out.println("^^^^^^^^^");
  }

  public ArrayList<String> predictPOSLabelsUsingBoW(ArrayList<String> inputs) {
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

  public ArrayList<String> predictPOSWLabelsUsingBoW(ArrayList<String> inputs) {
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

    Map<Pair<Integer, String>, Double> delta = new HashMap<Pair<Integer, String>, Double>();
    ArrayList<String> xs = new ArrayList<String>();
    xs.add("");
    xs.addAll(inputs);
    // base case
    System.out.println("---------");
    System.out.println(finalTheta);
    System.out.println("---------");
    for (String b : POSes) {
      // part1 = theta_e(b, xs.get(1))
      double part1 = finalTheta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(1)))];
      // part2 = theta_t(y0, b)
      double part2 = finalTheta[indexTransition(POSIndex.get(y0), POSIndex.get(b))];
      delta.put(new Pair<Integer, String>(1, b), Math.exp(part1 + part2));
    }
    // recurrence
    for (int k = 1; k < L; k++) {
      double newValue = 0;
      for (String b : POSes) {
        for (String a : POSes) {
          // part1 = theta_e(b, xs.get(k + 1))
          double part1 = finalTheta[indexEmission(POSIndex.get(b), wordIndex.get(xs.get(k + 1)))];
          // part2 = theta_t(a, b)
          double part2 = finalTheta[indexTransition(POSIndex.get(a), POSIndex.get(b))];
          double candidate = delta.get(new Pair<Integer, String>(k, a))*Math.exp(part1 + part2);
          if(candidate > newValue) {
            newValue = candidate;
          }
        }
        delta.put(new Pair<Integer, String>(k + 1, b), newValue);
      }
    }
    // backward
    ArrayList<String> ys = new ArrayList<String>();
    // we will add from the back and *reverse* the ys array at the end
    double maxValue = 0;
    String argmax = "";
    for(String b : POSes) {
      double candidate = delta.get(new Pair<Integer, String>(L, b));
      if(candidate > maxValue) {
        maxValue = candidate;
        argmax = b;
      }
    }
    ys.add(argmax);
    for(int i = L - 1; i >= 1; i++) {
      String yIPlusOneStar = ys.get(ys.size() - 1);
      maxValue = 0;
      argmax = "";
      for(String a : POSes) {
        double candidate =
          delta.get(new Pair<Integer, String>(i, a))
            *
          finalTheta[indexTransition(POSIndex.get(a), POSIndex.get(yIPlusOneStar))];
        if(candidate > maxValue) {
          maxValue = candidate;
          argmax = a;
        }
      }
      ys.add(argmax);
    }
    Collections.reverse(ys);
    return ys;
  }

  public ArrayList<String> predictPOSWLabels(ArrayList<String> inputs) {
    return inputs;
  }

  public void evaluate() {
    ArrayList<Pair<Integer, Integer>> statistics = new ArrayList<Pair<Integer, Integer>>();
    ArrayList<Pair<Integer, Integer>> statisticsW = new ArrayList<Pair<Integer, Integer>>();
    for(Sentence sentence : testSentences) {
      ArrayList<String> words = sentence.getAllX();
      ArrayList<String> predictions = predictPOSLabels(words);
      ArrayList<String> predictionsW = predictPOSWLabelsUsingBoW(words);
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
    public ArrayList<String> logZs;

    public Sentence() {
      tokens = new ArrayList<Token>();
      Xs = new ArrayList<String>();
      Ys = new ArrayList<String>();
      logZs = new ArrayList<String>();
    }

    public Sentence(ArrayList<Token> tokens) {
      this.tokens = tokens;
    }

    public void addToken(Token token) {
      tokens.add(token);
      Xs.add(token.x);
      Ys.add(token.y);
      logZs.add(token.z);
    }

    public Token getToken(int i) {
      return tokens.get(i);
    }

    public ArrayList<Token> getAllTokens(Token token) {
      return tokens;
    }

    public int length() {
      return tokens.size();
    }

    public ArrayList<String> getAllX() {
      return Xs;
    }

    public ArrayList<String> getAllY() {
      return Ys;
    }

    public ArrayList<String> getAllZ() {
      return logZs;
    }

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
