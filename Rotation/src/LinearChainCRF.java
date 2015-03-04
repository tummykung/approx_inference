import java.util.Arrays;
import java.util.Random;

import util.CartesianProduct;
import fig.basic.Option;
import fig.prob.SampleUtils;

public class LinearChainCRF {
  // input:
  double[][] edgeWeights, nodeWeights; // according to a HMM model
  private int approx_inference = 0;
  private int M; // the number of iterations in approximation inference
  //  Inference Type
  //  ==============
  //  0 = Exact
  //  1 = Approximate inference by
  //    - only sampling the model expectation
  //  2 = Importance sampling by
  //    - sampling the model expectation, using the importance weight
  //      to correct target sampling. However, still use exact inference
  //      to compute the importance weight.
  //  3 = Normalized importance sampling by
  //    - using the model weight as the proposal distribution
  private double xi; // relaxation parameter
  
  @Option(required=true)
  private boolean fully_supervised;
  @Option(required=false)
  private boolean learning_verbose;
  @Option(required=false)
  private boolean state_verbose;
  @Option(required=false)
  private boolean log_likelihood_verbose;
  @Option(required=false)
  private boolean prediction_verbose;
  @Option(required=false)
  private boolean sanity_check;

  // internal
  private long seed;
  private int d; // dimension of the parameters
  private int numX;
  private int numY;
  ForwardBackward my_model;
  
  public LinearChainCRF(
      long seed,
      double[][] edgeWeights,
      double[][] nodeWeights,
      double xi,
      int approx_inference,
      int M,
      boolean fully_supervised,
      boolean state_verbose,
      boolean log_likelihood_verbose,
      boolean prediction_verbose,
      boolean sanity_check
    ) {
    this.seed = seed;
    this.edgeWeights = edgeWeights; // Y x Y
    this.nodeWeights = nodeWeights; // Y x X
    this.xi = xi;
    this.approx_inference = 0;
    this.M = M;

    this.fully_supervised = fully_supervised;
    this.state_verbose = state_verbose;
    this.log_likelihood_verbose = log_likelihood_verbose;
    this.prediction_verbose = prediction_verbose;
    this.sanity_check = sanity_check;
    
    this.numX = edgeWeights.length;
    assert(edgeWeights.length == nodeWeights.length);
    this.numY = nodeWeights[0].length;
    this.d = this.numY * (this.numX + this.numY);

    my_model = new ForwardBackward(edgeWeights, nodeWeights);
  }
  
  public int[] sample(Random randomizer) {
    return my_model.sample(randomizer);
  }
  
  public void inferState() {
    my_model.infer();
  }

  private int indexTransition(int i, int j) {
    assert(0 <= i && i < numY);
    assert(0 <= j && j < numY);
    return numY*i + j;
  }

  private int indexEmission(int i, int j) {
    assert(0 <= i && i < numY);
    assert(0 <= j && j < numX);
    return numY*numY + numX*i + j;
  }

  public double[] phi(int[] z, int[] x) {
    double[] output = new double[d];
    for (int aIndex = 0; aIndex < numY; aIndex++) {
      for (int cIndex = 0; cIndex < numX; cIndex++) {
        output[indexEmission(aIndex, cIndex)] = 1;
      }
    }

    for (int aIndex = 0; aIndex < numY; aIndex++) {
      for (int bIndex = 0; bIndex < numY; bIndex++) {
        output[indexTransition(aIndex, bIndex)] = 1;
      }
    }
    return output;
  }

  public double getLogZ() {
    return my_model.getLogZ();
  }

}
