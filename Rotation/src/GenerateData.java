import java.util.ArrayList;

import util.Pair;
import fig.basic.LogInfo;
import fig.prob.SampleUtils;


public class GenerateData {
  

  public static ArrayList<AlignmentExample> generateDataSpeech(
      ArrayList<String> words
    ) throws Exception {

    ArrayList<AlignmentExample> examples = new ArrayList<AlignmentExample>();
    // generate a set A of lowercase English characters
    char A[] = new char['z'- 'a' + 1];
    int j = 0;
    for (char c='a'; c<='z'; c++)
      A[j++] = c;

    for (String y : words) {
      ArrayList<Character> ys = new ArrayList<Character>();
      for (char c : y.toCharArray()) {
        ys.add(c);
      }
      ArrayList<String> zs = new ArrayList<String>();
      for(int i = 0; i < y.length(); i++) {
        char c = y.charAt(i);
        // generate z
        zs.add("B-" + c);
        String z = "I-" + c;
        int num_additional_I_repeats = (int) SampleUtils.samplePoisson(Main.randomizer, Global.lambda1);
        for (int k = 0; k < num_additional_I_repeats; k++) {
          zs.add(z);
        }
        int num_additional_O_repeats = (int) SampleUtils.samplePoisson(Main.randomizer, Global.lambda2);
        for (int k = 0; k < num_additional_O_repeats; k++) {
          zs.add("O");
        }
      }

      // generate x
      ArrayList<Character> xs = new ArrayList<Character>(); 
      for (String z : zs) {
        char c;
        if(z.charAt(0) == 'B' || z.charAt(0) == 'I') {
          // B-c -> generate c with prob 1 - alpha, and uniform with prob alpha
          double prob = Main.randomizer.nextDouble();
          if(prob < Global.alpha) {
            c = z.charAt(2);
          } else {
            // select any character at random
            c = A[Main.randomizer.nextInt(A.length)];
          }
          xs.add(c);
        } else if (z.charAt(0) == 'O') {
          c = A[Main.randomizer.nextInt(A.length)];
          xs.add(c);
        }
      }
      AlignmentExample alignment = new AlignmentExample(xs, ys, zs);
      if (Main.debugVerbose) {
        LogInfo.logs(alignment.toStringHumanReadable());
      }
      examples.add(alignment);
    }

    return examples;
  }

  public static ArrayList<FullSupervisionExample> generateData(Params params) throws Exception {
    // return[n][0] = x is int[] and return[n][1] = y is also int[]
    // and n is the num sample iterator

    ArrayList<FullSupervisionExample> data;
    LogInfo.begin_track("generate_data");
    data = new ArrayList<FullSupervisionExample>();
    // given parameters, generate data according to a log-linear model
    // fully supervised:
    // p_{theta}(z | x) = 1/Z(theta; x) * exp(theta' * phi(x, z))
    //
    // full model:
    // p_{theta, xi} (y, z | x) = q_{xi}(y | z) * p_{theta}(z | x)
    // p_{theta}(z | x) = 1/Zp(theta; x) * exp(theta' * phi(x, z))
    // q_{xi}(y | z) = 1/Zq(xi; z) * exp(xi' * psi(y, z))

    // FIXME: generate this from some distribution. e.g., a uniform distribution
    // the things below are p(y) only, not p(y | x). but it's good enough
    // for testing emissions probability updates.
    
    // we will generate the x from a uniform distribution
    if (Main.sentenceLength <= 0) {
      throw new Exception("Please specify sentenceLength");
    }

    for (int n = 0; n < Main.numSamples; n++) {
      int[] x = new int[Main.sentenceLength];
      for(int i = 0; i < Main.sentenceLength; i++)
        x[i] = Util.randInt(0, Main.rangeX);

      ForwardBackward modelForEachX = ModelAndLearning.getForwardBackward(x, params);
      modelForEachX.infer();
      int[] seq = modelForEachX.sample(Main.randomizer);
      FullSupervisionExample example = new FullSupervisionExample(x, seq);
      data.add(example);
    }

    if(Main.stateVerbose) {
      System.out.println("done genetaing data.\nnum data instances: " + data.size());
      if(data.size() < 100) {
        System.out.println("data: " + data);
      }
    }
    LogInfo.end_track("generate_data");
    return data;
  }
}
