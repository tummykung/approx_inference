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

    StochasticGradientState state = new StochasticGradientState(
    	D,
    	dimPOS,
    	dimWords,
    	wordIndex,
    	POSIndex,
    	trainSentences,
    	y0,
    	POSes,
    	words
    );

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
        state.setExampleNum(i);
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
          if(candidate > newValue) {
            newValue = candidate;
          }
        }
        if (false) { // you can turn on once to see the values of logAlphas, but they're too noisy to print all the time
          System.out.println("logDelta = " + newValue);
        }
        logDeltaArray[k + 1][bIndex] = newValue;
      }
    }
    // backward: Viterbi
    ArrayList<String> ys = new ArrayList<String>();
    // we will add from the back and *reverse* the ys array at the end
    double maxValue = Double.NEGATIVE_INFINITY;
    String argmax = "";
    for(int bIndex = 0; bIndex < dimPOS; bIndex++) {
      double candidate = logDeltaArray[L][bIndex];
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

}
