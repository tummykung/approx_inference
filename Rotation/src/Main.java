/**
* Zero-one annealing experiment
* @author Tum Chaturapruek
*/

//import java.io.*;
//import java.util.*;
//import java.lang.*;
import java.util.ArrayList;
import java.util.Random;

import util.Pair;
import fig.basic.*;
import fig.exec.*;

public class Main implements Runnable {

    final public static int
        EXACT = 0,
        APPROX_SECOND_TERM = 1,
        MIXED_APPROX = 2,
        APPROX_BOTH_TERMS = 3;
    final public static int
        CONSTANT_STEP_SIZES = 0,
        DECREASING_STEP_SIZES = 1,
        ADAGRAD = 2;
    
    public static final int alphabetSize = 26;

    /** Inference Type
     * 0 =  Exact
     * 1 =  Approximate inference by only sampling the model expectation
     * 2 =  Importance sampling by sampling the model expectation, using the importance weight 
     *      to correct target sampling. However, still use exact inference
     *      to compute the importance weight.
     * 3 = Normalized importance sampling by using the model weight as the proposal distribution
     */
    @Option(required=true) public static long inferType;
    @Option(required=true) public static boolean fullySupervised;

    // (optional) flag for generating data
    @Option(required=false) public static boolean generateData;
    @Option(required=false) public static String dataSource = ""; // a path to data-set source
    @Option(required=false) public static String wordSource = ""; // a path to a list of words source

    // (optional) model parameters
    @Option(required=false) public static String experimentName = "SCRATCH";
    @Option(required=false) public static String model = "LinearChainCRF";
    @Option(required=false) public static double eta0 = 0.3; // gradient descent initial step size
    @Option(required=false) public static int gradientDescentType = ADAGRAD;
    @Option(required=false) public static long numIters = 10; // the number of samples in approximate inference
    @Option(required=false) public static double xi = 5.0; // the number of samples in approximate inference
    @Option(required=false) public static long seed;
    @Option(required=false) public static boolean usingAveragingMethod = true;

    // (optional) flags
    @Option(required=false) public static boolean learningVerbose = false;
    @Option(required=false) public static boolean stateVerbose = true;
    @Option(required=false) public static boolean debugVerbose = true;
    @Option(required=false) public static boolean extraVerbose = false;
    @Option(required=false) public static boolean logLikelihoodVerbose = true;
    @Option(required=false) public static boolean predictionVerbose = false;
    @Option(required=false) public static boolean sanityCheck = false;

    // for data generating
    @Option(required=false) public static int numSamples = 400;
    @Option(required=false) public static int sentenceLength;
    @Option(required=false) public static int rangeX = alphabetSize;
    @Option(required=false) public static int rangeY = alphabetSize;
    @Option(required=false) public static int rangeZ = 2 * alphabetSize + 1;

    public static Random randomizer;

    public static void main(String[] args) throws Exception {
      OptionsParser parser = new OptionsParser();
      parser.registerAll(
        new Object[]{
      });
      // Parse command line
      Execution.run(args, "Main", new Main(), parser);
    }

    @Override
    public void run() {
      try {
        runWithException();
      } catch(Exception e) {
        e.printStackTrace();
        throw new RuntimeException();
      }
    }

    void runWithException() throws Exception {
      // initialize randomizer
      randomizer = new Random(Main.seed);
      if (model.equals("LinearChainCRF")) {
        ModelAndLearning theModel = new ModelAndLearning();
        if (fullySupervised) {
          ArrayList<FullSupervisionExample> trainData = new ArrayList<FullSupervisionExample>();
          ArrayList<FullSupervisionExample> testData = new ArrayList<FullSupervisionExample>();

          Params trueParams = null; //only if we generate data using model features

          if (!generateData) {
            if(dataSource.equals("")) {
              String message = "If the flag generateData is false, then the source path must be specified.";
              throw new Exception(message);
            }
            Pair<ArrayList<FullSupervisionExample>, ArrayList<FullSupervisionExample>> trainAndTest = FullSupervisionReader.read(dataSource);
            trainData = trainAndTest.getFirst();
            testData = trainAndTest.getSecond();
            numSamples = trainData.size() + testData.size();
          } else {
            trueParams = new Params(Main.rangeX, Main.rangeY);

            for (int a = 0; a <= Main.rangeZ; a++)
              for (int b = 0; b <= Main.rangeZ; b++)
              {
                if(a == b) {
                  trueParams.transitions[a][b] = 15;
                } else if (Math.abs(a - b) <= 1) {
                  trueParams.transitions[a][b] = 5;
                } else {
                  trueParams.transitions[a][b] = 0;
                }
              }

            for (int a = 0; a <= Main.rangeZ; a++)
              for (int c = 0; c <= Main.rangeX; c++) {
                if(a == c) {
                  trueParams.emissions[a][c] = 20;
                } else if (Math.abs(a - c) <= 1) {
                  trueParams.emissions[a][c] = 5;
                } else {
                  trueParams.emissions[a][c] = 0;
                }
              }

            ArrayList<FullSupervisionExample> data = GenerateData.generateData(trueParams);
            double percentTrained = 0.80;
            int numTrained = (int) Math.round(percentTrained * data.size());
            for(int i = 0; i < data.size(); i++) {
              if(i < numTrained) {
                trainData.add(data.get(i));
              } else {
                testData.add(data.get(i));
              }
            }
          }

          if (debugVerbose) {
            System.out.println("----------BEGIN:trainData----------");
            for(Example example : trainData) {
              System.out.println(example);
            }
            System.out.println("----------END:trainData----------");
            System.out.println("----------BEGIN:testData----------");
            for(Example example : testData) {
              System.out.println(example);
            }
            System.out.println("----------END:testData----------");

            System.out.println("experimentName:\t" + experimentName);
            System.out.println("model:\t" + model);
            System.out.println("eta0:\t" + eta0);
            System.out.println("gradientDescentType:\t" + gradientDescentType);
            System.out.println("numIters:\t" + numIters);
            System.out.println("learningVerbose:\t" + learningVerbose);
            System.out.println("stateVerbose:\t" + stateVerbose);
            System.out.println("debugVerbose:\t" + debugVerbose);
            System.out.println("logLikelihoodVerbose:\t" + logLikelihoodVerbose);
            System.out.println("predictionVerbose:\t" + predictionVerbose);
            System.out.println("sanityCheck:\t" + sanityCheck);
            System.out.println("num_samples:\t" + numSamples);
            System.out.println("sentenceLength:\t" + sentenceLength);
            System.out.println("rangeX:\t" + rangeX);
            System.out.println("rangeY:\t" + rangeY);
            System.out.println("rangeZ:\t" + rangeZ);
          }

          Params learnedParams = theModel.trainFullSupervision(trainData);
          learnedParams.print("learnedParams");
          
          if(Main.generateData && trueParams != null) {
            trueParams.print("trueParams");
          }
          LogInfo.begin_track("trainReport");
            Report learnerTrainReport = theModel.testFullSupervision(trainData, learnedParams);
            
            LogInfo.begin_track("learner");
            learnerTrainReport.print("train.learner");
            LogInfo.end_track("learner");
            
            if(Main.generateData && trueParams != null) {
              LogInfo.begin_track("expert");
              Report expertTrainReport = theModel.testFullSupervision(trainData, trueParams);
              expertTrainReport.print("train.expert");
              LogInfo.end_track("expert");
            }
          LogInfo.end_track("trainReport");

          LogInfo.begin_track("testReport");
            LogInfo.begin_track("learner");
            Report learnerReport = theModel.testFullSupervision(testData, learnedParams);
            learnerReport.print("test.learner");
            LogInfo.end_track("learner");
    
            if(Main.generateData && trueParams != null) {
                LogInfo.begin_track("expert");
                Report expertReport = theModel.testFullSupervision(testData, trueParams);
                expertReport.print("test.expert");
                LogInfo.end_track("expert");
            }
          LogInfo.end_track("testReport");
        } else {
          // indirectly supervised
          ArrayList<AlignmentExample> trainData = new ArrayList<AlignmentExample>();
          ArrayList<AlignmentExample> testData = new ArrayList<AlignmentExample>();
          
          Global.lambda1 = 1.0;
          Global.lambda2 = 0.3;
          Global.alpha = 0.9;
          ArrayList<String> words = WordReader.read(wordSource);
//          ArrayList<String> words = new ArrayList<String>();
//          words.add("test");
//          words.add("what");
          ArrayList<AlignmentExample> data = GenerateData.generateDataSpeech(words);
          double percentTrained = 0.80;
          int numTrained = (int) Math.round(percentTrained * data.size());
          for(int i = 0; i < data.size(); i++) {
            if(i < numTrained) {
              trainData.add(data.get(i));
            } else {
              testData.add(data.get(i));
            }
          }
          
          if (debugVerbose) {
            System.out.println("----------BEGIN:trainData----------");
            for(Example example : trainData) {
              System.out.println(example);
            }
            System.out.println("----------END:trainData----------");
            System.out.println("----------BEGIN:testData----------");
            for(Example example : testData) {
              System.out.println(example);
            }
            System.out.println("----------END:testData----------");

            System.out.println("experimentName:\t" + experimentName);
            System.out.println("model:\t" + model);
            System.out.println("eta0:\t" + eta0);
            System.out.println("gradientDescentType:\t" + gradientDescentType);
            System.out.println("numIters:\t" + numIters);
            System.out.println("learningVerbose:\t" + learningVerbose);
            System.out.println("stateVerbose:\t" + stateVerbose);
            System.out.println("debugVerbose:\t" + debugVerbose);
            System.out.println("logLikelihoodVerbose:\t" + logLikelihoodVerbose);
            System.out.println("predictionVerbose:\t" + predictionVerbose);
            System.out.println("sanityCheck:\t" + sanityCheck);
            System.out.println("num_samples:\t" + numSamples);
            System.out.println("sentenceLength:\t" + sentenceLength);
            System.out.println("rangeX:\t" + rangeX);
            System.out.println("rangeY:\t" + rangeY);
            System.out.println("rangeZ:\t" + rangeZ);
          }

          Params learnedParams = theModel.trainIndirectSupervision(trainData);
          learnedParams.print("learnedParams");

          LogInfo.begin_track("trainReport");
            Report learnerTrainReport = theModel.testIndirectSupervision(trainData, learnedParams);
            
            LogInfo.begin_track("learner");
            learnerTrainReport.print("train.learner");
            LogInfo.end_track("learner");
          LogInfo.end_track("trainReport");

          LogInfo.begin_track("testReport");
            LogInfo.begin_track("learner");
            Report learnerReport = theModel.testIndirectSupervision(testData, learnedParams);
            learnerReport.print("test.learner");
            LogInfo.end_track("learner");

          LogInfo.end_track("testReport");
        }

      } else {
        throw new Exception("Model not supported");
      }
    }
}
