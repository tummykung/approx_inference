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

    @Option(required=true)
    public static long inferType;
    //  Inference Type
    //  ==============
    //  0 = Exact
    //  1 = Approximate inference by
    //    - only sampling the model expectation
    //  2 = Importance sampling by
    //    - sampling the model expectation, using the importance weight
    //      to correct target sampling. However, still use exact inference
    //      to compute the importance weight.
    //  3 = Normalized importance sampling by
    //    - using the model weight as the proposal distribution
    @Option(required=true)
    public static boolean fully_supervised; // can switch to fully supervised for sanity check

    // (optional) flag for generating data
    @Option(required=false)
    public static boolean generate_data;
    @Option(required=false)
    public static String datasource = ""; // a path to data-set source

    // (optional) model parameters
    @Option(required=false)
    public static String experimentName = "SCRATCH";
    @Option(required=false)
    public static String model = "LinearChainCRF";
    @Option(required=false)
    public static double eta0 = 0.3; // gradient descent initial step size
    @Option(required=false)
    public static int gradient_descent_type = ADAGRAD;
    @Option(required=false)
    public static long numiters = 10; // the number of samples in approximate inference
    @Option(required=false)
    public static double xi = 10.0; // the number of samples in approximate inference
    @Option(required=false)
    public static long seed = 12345671;

    // (optional) flags
    @Option(required=false)
    public static boolean learning_verbose = false;
    @Option(required=false)
    public static boolean state_verbose = true;
    @Option(required=false)
    public static boolean debug_verbose = true;
    @Option(required=false)
    public static boolean extra_verbose = false;
    @Option(required=false)
    public static boolean log_likelihood_verbose = true;
    @Option(required=false)
    public static boolean prediction_verbose = false;
    @Option(required=false)
    public static boolean sanity_check = false;
    

    // for data generating
    @Option(required=false)
    public static int num_samples = 400;
    @Option(required=false)
    public static int sentenceLength = 5;
    @Option(required=false)
    public static int rangeX = 5;
    @Option(required=false)
    public static int rangeY = 5;
    @Option(required=false)
    public static int rangeZ = 5;

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
        ModelAndLearning the_model = new ModelAndLearning();
        
        ArrayList<Example> train_data = new ArrayList<Example>();
        ArrayList<Example> test_data = new ArrayList<Example>();

        if (!generate_data) {
          if(datasource.equals("")) {
            String message = "If the flag generate_data is false, then the source path must be specified.";
            throw new Exception(message);
          }
          Pair<ArrayList<Example>, ArrayList<Example>> train_and_test = Reader.read(datasource);
          train_data = train_and_test.getFirst();
          test_data = train_and_test.getSecond();
          num_samples = train_data.size() + test_data.size();
        } else {
          ArrayList<Example> data = the_model.generate_data();
          double percent_trained = 0.80;
          int num_trained = (int) Math.round(percent_trained * data.size());
          for(int i = 0; i < data.size(); i++) {
            if(i < num_trained) {
              train_data.add(data.get(i));
            } else {
              test_data.add(data.get(i));
            }
          }
        }

        if (debug_verbose) {
          System.out.println("----------BEGIN:train_data----------");
          for(Example example : train_data) {
            System.out.println(example);
          }
          System.out.println("----------END:train_data----------");
          System.out.println("----------BEGIN:test_data----------");
          for(Example example : test_data) {
            System.out.println(example);
          }
          System.out.println("----------END:test_data----------");


          System.out.println("experimentName:\t" + experimentName);
          System.out.println("model:\t" + model);
          System.out.println("eta0:\t" + eta0);
          System.out.println("gradient_descent_type:\t" + gradient_descent_type);
          System.out.println("numiters:\t" + numiters);
          System.out.println("learning_verbose:\t" + learning_verbose);
          System.out.println("state_verbose:\t" + state_verbose);
          System.out.println("debug_verbose:\t" + debug_verbose);
          System.out.println("log_likelihood_verbose:\t" + log_likelihood_verbose);
          System.out.println("prediction_verbose:\t" + prediction_verbose);
          System.out.println("sanity_check:\t" + sanity_check);
          System.out.println("num_samples:\t" + num_samples);
          System.out.println("sentenceLength:\t" + sentenceLength);
          System.out.println("rangeX:\t" + rangeX);
          System.out.println("rangeY:\t" + rangeY);
          System.out.println("rangeZ:\t" + rangeZ);
        }
        the_model.train(train_data);
      } else {
        throw new Exception("Model not supported");
      }
    }
}
