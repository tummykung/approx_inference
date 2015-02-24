/**
* Zero-one annealing experiment
* @author Tum Chaturapruek
*/

//import java.io.*;
//import java.util.*;
//import java.lang.*;
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
        //final String outputFile = argMap.containsKey("-outputAlignments") ? argMap.get("-outputAlignments") : "";
        //String dataset = argMap.containsKey("-evalSet") ? argMap.get("-evalSet") : "miniTest";
        //if (outputFile.length() > 0) dataset = "";
        //String basePath = argMap.containsKey("-dataPath") ? argMap.get("-dataPath") : DATA_PATH;
        //basePath += dataset.equalsIgnoreCase("miniTest") ? "/mini" : "/"+language;

        if (model.equals("LinearChainCRF")) {
        	LinearChainCRF the_model = new LinearChainCRF(seed);
        	double[] parameters = {3.00, 3.00, 3.00, 4.00};
        	the_model.generate_data(parameters);
        } else {
        	throw new Exception("Model not supported");
        }
        
    }
}
