import java.util.ArrayList;

import util.Pair;

public class Reader {
  public static Pair<ArrayList<Example>, ArrayList<Example>> read(String source) throws Exception {
    ArrayList<Example> trainData = new ArrayList<Example>();
    ArrayList<Example> testData = new ArrayList<Example>();

    String state = "ground";
    String subState = "ground";
    Example example = new Example(null, null); // will be half-built after seeing an input
    for (String line : fig.basic.IOUtils.readLinesHard(source)) {
      if (Main.debugVerbose) {
        System.out.println(line);
      }
      if(line.equals("----------BEGIN:trainData----------")) {
        if(!state.equals("ground"))
          throw new Exception("ill-formed data. Must be in the ground state before begin");
        state = "begin:trainData";
        continue;
      } else if (line.equals("----------END:trainData----------")) {
        if(!state.equals("begin:trainData"))
          throw new Exception("ill-formed data. Must begin before end");
        state = "ground";
        continue;
      }
      if(line.equals("----------BEGIN:testData----------")) {
        if(!state.equals("ground"))
          throw new Exception("ill-formed data. Must be in the ground state before begin");
        state = "begin:testData";
        continue;
      } else if (line.equals("----------END:testData----------")) {
        if(!state.equals("begin:testData"))
          throw new Exception("ill-formed data. Must begin before end");
        state = "ground";
        continue;
      }
      if (state.equals("begin:trainData") || state.equals("begin:testData")) {
        String[] things = line.split(" ");
        if(things[0].equals("input") && subState.equals("ground")) {
          subState = "input";
          int[] input = new int[things.length - 1];
          for(int i = 0; i < things.length - 1; i++) {
            input[i] = Integer.parseInt(things[i + 1]);
          }
          example = new Example(input, null);
        } else if (things[0].equals("output") && subState.equals("input")) {
          subState = "ground";
          int[] output = new int[things.length - 1];
          for(int i = 0; i < things.length - 1; i++) {
            output[i] = Integer.parseInt(things[i + 1]);
          }
          example.setOutput(output);
          if(state.equals("begin:trainData")) {
            trainData.add(example);
          } else if (state.equals("begin:testData")) {
            testData.add(example);
          }
        }
      }
    }
    return new Pair<ArrayList<Example>, ArrayList<Example>>(trainData, testData);
  }
}
