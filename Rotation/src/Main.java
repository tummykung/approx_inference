/**
* Zero-one annealing experiment
* @author Tum Chaturapruek
*/

//import java.io.*;
//import java.util.*;
//import java.lang.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import util.CommandLineUtils;
import util.Pair;
import fig.basic.*;
import fig.exec.*;

public class Main implements Runnable {
    public final static long DEFAULT_SEED = 989181171;

    final public static int
        EXACT = 0,
        APPROX_SECOND_TERM = 1,
        MIXED_APPROX = 2,
        APPROX_BOTH_TERMS = 3;
    final public static int
        CONSTANT_STEP_SIZES = 0,
        DECREASING_STEP_SIZES = 1,
        ADAGRAD = 2;

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
    @Option(required=false) public static String datasource = ""; // a path to data-set source

    // (optional) model parameters
    @Option(required=false) public static String experimentName = "SCRATCH";
    @Option(required=false) public static String model = "LinearChainCRF";
    @Option(required=false) public static double eta0 = 0.3; // gradient descent initial step size
    @Option(required=false) public static int gradientDescentType = ADAGRAD;
    @Option(required=false) public static long numIters = 10; // the number of samples in approximate inference
    @Option(required=false) public static double xi = 10.0; // the number of samples in approximate inference
    @Option(required=false) public static long seed = 12345671;
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
    @Option(required=false) public static int rangeX = 5;
    @Option(required=false) public static int rangeY = 5;
    @Option(required=false) public static int rangeZ = 5;

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
      if (model.equals("LinearChainCRF")) {
        ModelAndLearning theModel = new ModelAndLearning();
        
        ArrayList<Example> trainData = new ArrayList<Example>();
        ArrayList<Example> testData = new ArrayList<Example>();
        Params trueParams = null; //only if we generate data using model features

        if (!generateData) {
          if(datasource.equals("")) {
            String message = "If the flag generateData is false, then the source path must be specified.";
            throw new Exception(message);
          }
          Pair<ArrayList<Example>, ArrayList<Example>> trainAndTest = Reader.read(datasource);
          trainData = trainAndTest.getFirst();
          testData = trainAndTest.getSecond();
          numSamples = trainData.size() + testData.size();
        } else {
          Pair<ArrayList<Example>, Params> dataAndParams = theModel.generateData();
          ArrayList<Example> data = dataAndParams.getFirst();
          trueParams = dataAndParams.getSecond();
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
        Params learnedParams = theModel.train(trainData);
        learnedParams.print("learnedParams");
        
        if(Main.generateData && trueParams != null) {
          trueParams.print("trueParams");
        }
        LogInfo.begin_track("trainReport");
          Report learnerTrainReport = theModel.test(trainData, learnedParams);
          
          LogInfo.begin_track("learner");
          learnerTrainReport.print("learner");
          LogInfo.end_track("learner");
          
          if(Main.generateData && trueParams != null) {
            LogInfo.begin_track("expert");
            Report expertTrainReport = theModel.test(trainData, trueParams);
            expertTrainReport.print("expert");
            LogInfo.end_track("expert");
          }
        LogInfo.end_track("trainReport");


        LogInfo.begin_track("testReport");
          LogInfo.begin_track("learner");
          Report learnerReport = theModel.test(testData, learnedParams);
          learnerReport.print("learner");
          LogInfo.end_track("learner");
  
          if(Main.generateData && trueParams != null) {
              LogInfo.begin_track("expert");
              Report expertReport = theModel.test(testData, trueParams);
              expertReport.print("expert");
              LogInfo.end_track("expert");
          }
        LogInfo.end_track("testReport");
      } else {
        throw new Exception("Model not supported");
      }
    }
}
