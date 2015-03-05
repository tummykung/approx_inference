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

    
    // (optional) flag for generating data
    @Option(required=false)
    public static boolean generate_data;
    @Option(required=false)
    public static String dataset = ""; // a path to dataset

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
        int num_samples = 100;
        int sentenceLength = 5;
        ArrayList<Example> train_data = new ArrayList<Example>();
        ArrayList<Example> test_data = new ArrayList<Example>();

        if (!generate_data) {
          if(dataset.equals("")) {
            String message = "If the flag generate_data is false, then the dataset path must be specified.";
            throw new Exception(message);
          }
          // TODO: refactor this to a Reader class
          String state = "ground";
          String sub_state = "ground";
          Example example = new Example(null, null); // will be half-built after seeing an input
          for (String line : fig.basic.IOUtils.readLinesHard(dataset)) {
            System.out.println(line);
            if(line.equals("----------BEGIN:train_data----------")) {
              if(!state.equals("ground"))
                throw new Exception("ill-formed data. Must be in the ground state before begin");
              state = "begin:train_data";
              continue;
            } else if (line.equals("----------END:train_data----------")) {
              if(!state.equals("begin:train_data"))
                throw new Exception("ill-formed data. Must begin before end");
              state = "ground";
              continue;
            }
            if(line.equals("----------BEGIN:test_data----------")) {
              if(!state.equals("ground"))
                throw new Exception("ill-formed data. Must be in the ground state before begin");
              state = "begin:test_data";
              continue;
            } else if (line.equals("----------END:test_data----------")) {
              if(!state.equals("begin:test_data"))
                throw new Exception("ill-formed data. Must begin before end");
              state = "ground";
              continue;
            }
            if (state.equals("begin:train_data") || state.equals("begin:test_data")) {
              String[] things = line.split(" ");
              if(things[0].equals("input") && sub_state.equals("ground")) {
                sub_state = "input";
                int[] input = new int[things.length - 1];
                for(int i = 0; i < things.length - 1; i++) {
                  input[i] = Integer.parseInt(things[i + 1]);
                }
                example = new Example(input, null);
              } else if (things[0].equals("output") && sub_state.equals("input")) {
                sub_state = "ground";
                int[] output = new int[things.length - 1];
                for(int i = 0; i < things.length - 1; i++) {
                  output[i] = Integer.parseInt(things[i + 1]);
                }
                example.setOutput(output);
                if(state.equals("begin:train_data")) {
                  train_data.add(example);
                } else if (state.equals("begin:test_data")) {
                  test_data.add(example);
                }
              }
            }
          }
        } else {
          ArrayList<Example> data = the_model.generate_data(num_samples, sentenceLength);
          double percent_trained = 0.80;
          int num_trained = (int)(percent_trained * data.size());
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
        }

        the_model.train(train_data);
        System.out.println("the_end");
      } else {
        throw new Exception("Model not supported");
      }
    }
}
