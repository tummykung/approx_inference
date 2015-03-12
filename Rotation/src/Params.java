import java.util.ArrayList;

import fig.basic.Fmt;
import fig.basic.LogInfo;


public class Params {
  public double[][] transitions;
  public double[][] emissions;
  // TODO: generalize this to phi(x) \otimes Y
  
  ArrayList<FeatureExtractor> features;
  double xi;
  
  public Params(int rangeX, int rangeY, double initialValue) {
    transitions = new double[rangeY + 1][rangeY + 1]; // from left to right
    emissions = new double[rangeY + 1][rangeX + 1];
    
    if (initialValue != 0) {
      for (int a = 0; a <= rangeY; a++)
        for (int b = 0; b <= rangeY; b++)
          transitions[a][b] = initialValue;
          
    for (int a = 0; a <= rangeY; a++)
      for (int c = 0; c <= rangeX; c++)
        emissions[a][c] = initialValue;
    }
  }
  public Params(int rangeX, int rangeY) {
    this(rangeX, rangeY, 0);
  }
  
  public void print(String name) {
    LogInfo.logs(name + ".emissions =\n" + Fmt.D(emissions, "\n"));
    LogInfo.logs(name + ".transitions =\n" + Fmt.D(transitions, "\n"));
  }
}
