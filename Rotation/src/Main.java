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
        DECREASING_STEP_SIZES = 1;

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

    // (optional) model parameters
    @Option(required=false)
    public static String experimentName = "SCRATCH";
    @Option(required=false)
    public static String model = "LinearChainCRF";
    @Option(required=false)
    public static double eta0 = 0.01; // gradient descent initial step size
    @Option(required=false)
    public static int gradient_descent_type = 0;
    @Option(required=false)
    public static long M = 10; // the number of samples in approximate inference
    @Option(required=false)
    public static double xi = 10.0; // the number of samples in approximate inference
    @Option(required=false)
    public static long seed = 12345671;

    // (optional) flags
    public static boolean learning_verbose = false;
    @Option(required=false)
    public static boolean state_verbose = true;
    @Option(required=false)
    public static boolean debug_verbose = true;
    @Option(required=false)
    public static boolean log_likelihood_verbose = true;
    @Option(required=false)
    public static boolean prediction_verbose = false;
    @Option(required=false)
    public static boolean sanity_check = false;


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
        int num_samples = 1000;
        int sentenceLength = 5;
        ArrayList<Example> data = the_model.generate_data(num_samples, sentenceLength);
        double percent_trained = 0.80;
        int num_trained = (int)(percent_trained * data.size());
        ArrayList<Example> train_data = new ArrayList<Example>();
        ArrayList<Example> test_data = new ArrayList<Example>();
        for(int i = 0; i < num_trained; i++) {
          if(i < num_trained) {
            train_data.add(data.get(i));
          } else {
            test_data.add(data.get(i));
          }
        }

        if (debug_verbose) {
          System.out.println("train_data: " + train_data);
          System.out.println("test_data: " + test_data);
        }

        the_model.train(train_data);
        System.out.println("the_end");
      } else {
        throw new Exception("Model not supported");
      }
    }
}
