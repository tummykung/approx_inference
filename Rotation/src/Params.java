import java.util.ArrayList;


public class Params {
  public double[][] transitions;
  public double[][][] emissions;
//  double[][][] emissions; // [featureTemplateIndex][y][featureTemplateValue]
  // phi(x) \otimes Y
  
  ArrayList<FeatureExtractor> features;
  double xi;
  
  public Params(int X, int Y, int featureTemplateNum) {
    transitions = new double[Y][Y];
    emissions = new double[featureTemplateNum][Y][X];
  }
}
