import java.util.ArrayList;

import fig.basic.Fmt;
import fig.basic.LogInfo;


public class Params {
  public double[][] transitions;
  public double[][] emissions;
//  double[][][] emissions; // [featureTemplateIndex][y][featureTemplateValue]
  // phi(x) \otimes Y
  
  ArrayList<FeatureExtractor> features;
  double xi;
  
  public Params(int X, int Y, int featureTemplateNum, double initial_value) {
    transitions = new double[Y][Y]; // from left to right
    emissions = new double[Y][X];
    
    if (initial_value != 0) {
      for (int a = 0; a < Y; a++)
        for (int b = 0; b < Y; b++)
          transitions[a][b] = initial_value;
          
    for (int a = 0; a < Y; a++)
      for (int c = 0; c < X; c++)
        emissions[a][c] = initial_value;
    }
  }
  public Params(int X, int Y, int featureTemplateNum) {
    this(X, Y, featureTemplateNum, 0);
  }
  
  public void print(String name) {
    LogInfo.logs(name + ".emissions =\n" + Fmt.D(emissions, "\n"));
    LogInfo.logs(name + ".transitions =\n" + Fmt.D(transitions, "\n"));
  }
}
