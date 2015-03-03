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
          // N = Length of sequence (sentenceLength)
          // S = Number of states

          // elements selected from Y times Y 
          // dimension is S x S
          double [][] edgeWeights = new double[][]{
              {1, 10, 1},
              {3, 4, 1},
              {1, 2, 3}
           };

          // elements selected from X times Y
          // dimension is N x S
          double [][] nodeWeights = new double[][]{
              {10, 2, 3},   // at x0
              {4, 10, 4},   // at x1
              {4, 3, 20},   // at x2
              {5, 2, 1}     // at x3
          };
          double xi = 10.0;
          int approx_inference = 0;
          int M = 10;
          LinearChainCRF the_model = new LinearChainCRF(
              seed,
              edgeWeights,
              nodeWeights,
              xi,
              approx_inference,
              M
          );
          int num_samples = 1000;
          int[][][] data = the_model.generate_data(num_samples);
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
