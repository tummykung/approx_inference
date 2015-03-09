//import java.util.Arrays;
//import java.util.Random;
//
//import fig.basic.Option;
//import fig.prob.SampleUtils;
//
//public class LinearChainCRF {
//  // input:
//  double[][] edgeWeights, nodeWeights; // according to a HMM model 
//
//  // internal
//  ForwardBackward my_model;
//  
//  public LinearChainCRF(
//      double[][] edgeWeights,
//      double[][] nodeWeights
//    ) {
//    this.edgeWeights = edgeWeights; // dimension of Y x Y
//    this.nodeWeights = nodeWeights; // dimension of Y x X
//    my_model = new ForwardBackward(edgeWeights, nodeWeights);
//  }
//  
//  public int[] sample(Random randomizer) {
//    return my_model.sample(randomizer);
//  }
//  
//  public void inferState(Example example, Params param) {
//    my_model.infer();
//  }
//
//  public void forwardBackwardInfer() {
//    my_model.infer();
//  }
//
//  public double getLogZ() {
//    return my_model.getLogZ();
//  }
//
//}
