import java.util.ArrayList;

import fig.basic.Fmt;
import fig.basic.LogInfo;


public class Params {
  public double[][] transitions;
  public double[][][] emissions;
//  double[][][] emissions; // [featureTemplateIndex][y][featureTemplateValue]
  // phi(x) \otimes Y
  
  ArrayList<FeatureExtractor> features;
  double xi;
  
  public Params(int X, int Y, int featureTemplateNum) {
    transitions = new double[Y][Y]; // from left to right
    emissions = new double[featureTemplateNum][Y][X];
  }
  
  public void print(String name) {
    LogInfo.logs(name + ".emissions = ");
    int L = emissions.length;
    for(int i = 0; i < L; i++) {
      LogInfo.logs(">> i = " + i + ":");
      LogInfo.logs(Fmt.D(emissions[i], "\n"));
    }
    LogInfo.logs(name + ".transitions =\n" + Fmt.D(transitions, "\n"));
  }
}
