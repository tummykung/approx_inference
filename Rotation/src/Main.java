/**
* Zero-one annealing experiment
* @author Tum Chaturapruek
*/

//import java.io.*;
//import java.util.*;
//import java.lang.*;
import java.util.Arrays;
import java.util.Map;

import util.CommandLineUtils;
import fig.basic.*;
import fig.exec.*;

public class Main {
    public LinearChainCRF model;
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
    private static boolean fully_supervised = true; // can switch to fully supervised for sanity check
    @Option(required=false)
    private static boolean learning_verbose = true;
    @Option(required=false)
    private static boolean state_verbose = true;
    @Option(required=false)
    private static boolean log_likelihood_verbose = true;
    @Option(required=false)
    private static boolean prediction_verbose = false;
    @Option(required=false)
    private static boolean sanity_check = false;

    
    public static void main(String[] args) throws Exception {
        // Parse command line
        final Map<String,String> argMap = CommandLineUtils.simpleCommandLineParser(args);
        final long seed = argMap.containsKey("-seed") ? 
            Long.parseLong(argMap.get("-seed")) : DEFAULT_SEED;
        final boolean verbose = argMap.containsKey("-verbose");
        final String model = argMap.containsKey("-model") ? 
            argMap.get("-model") : "LinearChainCRF";
        final String experimentName = argMap.containsKey("-experimentName") ? 
                    argMap.get("-experimentName") : "SCRATCH";

        if (model.equals("LinearChainCRF")) {
          ModelAndLearning the_model = new ModelAndLearning(
              seed,
              fully_supervised,
              state_verbose,
              log_likelihood_verbose,
              prediction_verbose,
              sanity_check
          );
          int num_samples = 1000;
          int sentenceLength = 5;
          int[][][] data = the_model.generate_data(num_samples, sentenceLength);
          double percent_trained = 0.80;
          int num_trained = (int)(percent_trained * data.length);
          int[][][] train_data = Arrays.copyOfRange(data, 0, num_trained);

          int[][][] test_data = Arrays.copyOfRange(data, num_trained, data.length);
          if (verbose) {
            System.out.println("train_data: " + Arrays.deepToString(train_data));
            System.out.println("test_data: " + Arrays.deepToString(test_data));
          }

          the_model.train(train_data);
          System.out.println("the_end");
        } else {
          throw new Exception("Model not supported");
        }
        
    }
}
