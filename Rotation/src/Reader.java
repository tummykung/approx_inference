import java.util.ArrayList;

import util.Pair;

public class Reader {
  public static Pair<ArrayList<Example>, ArrayList<Example>> read(String source) throws Exception {
    ArrayList<Example> train_data = new ArrayList<Example>();
    ArrayList<Example> test_data = new ArrayList<Example>();

    String state = "ground";
    String sub_state = "ground";
    Example example = new Example(null, null); // will be half-built after seeing an input
    for (String line : fig.basic.IOUtils.readLinesHard(source)) {
      if (Main.debug_verbose) {
        System.out.println(line);
      }
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
    return new Pair<ArrayList<Example>, ArrayList<Example>>(train_data, test_data);
  }
}
