import java.util.Arrays;
import java.util.Random;

import fig.basic.Option;
import fig.prob.SampleUtils;

public class LinearChainCRF {
  // input:
  double[][] edgeWeights, nodeWeights; // according to a HMM model 

  // internal
  ForwardBackward my_model;
  
  public LinearChainCRF(
      double[][] edgeWeights,
      double[][] nodeWeights
    ) {
    this.edgeWeights = edgeWeights; // Y x Y
    this.nodeWeights = nodeWeights; // Y x X
    my_model = new ForwardBackward(edgeWeights, nodeWeights);
  }
  
  public int[] sample(Random randomizer) {
    return my_model.sample(randomizer);
  }
  
  public void inferState() {
    my_model.infer();
  }

  public double[] phi(int[] z, int[] x) {
    double[] output = new double[1]; // TODO: fix this
    
//    for (int aIndex = 0; aIndex < numY; aIndex++) {
//      for (int cIndex = 0; cIndex < numX; cIndex++) {
//        output[indexEmission(aIndex, cIndex)] = 1;
//      }
//    }
//
//    for (int aIndex = 0; aIndex < numY; aIndex++) {
//      for (int bIndex = 0; bIndex < numY; bIndex++) {
//        output[indexTransition(aIndex, bIndex)] = 1;
//      }
//    }
    return output;
  }

  public double getLogZ() {
    return my_model.getLogZ();
  }

}
