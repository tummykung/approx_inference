import java.util.ArrayList;
import java.util.Map;

import fig.basic.Maximizer;
public class StochasticGradientState implements Maximizer.FunctionState {
  // Current theta, value, gradient
  private double[] theta;
  private double value;
  private double[] gradient;
  private boolean valueValid = false;
  private boolean gradientValid = false;
  private boolean cacheValid = false;
  private static final boolean DEBUG = false;
  private Double[][] logAlphaArray;
  private Double[][] logBetaArray;
  private double logZ;
  private int sentenceCounter = 0;
  private int D, dimPOS, dimWords;
  private Map<String,Integer> wordIndex;
  private Map<String,Integer> POSIndex;
  private ArrayList<Sentence> trainSentences;
  private String y0;
  private ArrayList<String> POSes;
  private ArrayList<String> words;

  public void setExampleNum(int i){
    sentenceCounter = i;
  }

  public StochasticGradientState(
	int D,
	int dimPOS,
	int dimWords, 
	Map<String,Integer> wordIndex,
	Map<String,Integer> POSIndex,
	ArrayList<Sentence> trainSentences,
	String y0,
	ArrayList<String> POSes,
	ArrayList<String> words
  ){
    this.D = D;
    this.dimPOS = dimPOS;
    this.dimWords = dimWords;
    this.wordIndex = wordIndex;
    this.POSIndex = POSIndex;
    this.trainSentences = trainSentences;
    this.y0 = y0;
    this.POSes = POSes;
    this.words = words;
    theta = new double[D];
    gradient = new double[D];
  }
  public int indexTransition(int i, int j) {
	  assert(0 <= i && i < dimPOS);
	  assert(0 <= j && j < dimPOS);
	  return dimPOS*i + j;
  }

  public int indexEmission(int i, int j) {
	  assert(0 <= i && i < dimPOS);
	  assert(0 <= j && j < dimWords);
	  return dimPOS*dimPOS + dimWords*i + j;
  }

  public double[] point() { return theta; }

  public void calculateCachesSentence(Sentence sentence) {
    // start these arrays fresh!
      logAlphaArray = new Double[sentence.length() + 1][dimPOS];
      logBetaArray = new Double[sentence.length() + 1][dimPOS];

      ArrayList<String> xs = new ArrayList<String>();
      xs.add(""); // it doesn't really matter what x0 is. We just want to align xs with ys.
      xs.addAll(sentence.getAllX());
      ArrayList<String> ys = new ArrayList<String>();
      ys.add(y0);
      ys.addAll(sentence.getAllY());

      // ---------------------- computing logAlphas -------------------
      // base case
      for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
        // part1 = theta_e(b, xs.get(1))
        double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(1)))];
        // part2 = theta_t(y0, b)
        double part2 = theta[indexTransition(POSIndex.get(y0), bIndex)];
        logAlphaArray[1][bIndex] = part1 + part2;
      }
      // recurrence
      for (int k = 1; k <= sentence.length() - 1; k++) {
        for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
          ArrayList<Double> exponents = new ArrayList<Double>();
          for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
            // part1 = theta_e(b, xs.get(k + 1))
            double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(k + 1)))];
            // part2 = theta_t(a, b)
            double part2 = theta[indexTransition(aIndex, bIndex)];
            double part3 = logAlphaArray[k][aIndex];
            exponents.add(part1 + part2 + part3);
          }

          // use the log sum exp trick to stabilize numerics:
          // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
          double maxExponent = Double.NEGATIVE_INFINITY;
          for(double exponent : exponents) {
            if(exponent > maxExponent) {
              maxExponent = exponent;
            }
          }
          double newValue = 0;
          for(double exponent : exponents) {
            newValue += Math.exp(exponent - maxExponent);
          }
          double newLogAlpha = maxExponent + Math.log(newValue);
          if (DEBUG) {
            System.out.println("newLogAlpha = " + newLogAlpha);
          }
          logAlphaArray[k + 1][bIndex] = newLogAlpha;
        }
      }

      // ---------------------- computing logBetas -------------------
      // base case
      for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
        logBetaArray[sentence.length()][aIndex] = 0.00;
      }

      // recurrence
      for (int k = sentence.length(); k >= 1; k--) {
        double newValue = 0;

        if (k > 1) {
          for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
            ArrayList<Double> exponents = new ArrayList<Double>();
            for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
              // part1 = theta_e(a, xs.get(k))
              double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(k)))];
              // part2 = theta_t(a, b)
              double part2 = theta[indexTransition(aIndex, bIndex)];
              double part3 = logBetaArray[k][bIndex];
              exponents.add(part1 + part2 + part3);
            }

            // use the log sum exp trick to stabilize numerics:
            // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
            double maxExponent = Double.NEGATIVE_INFINITY;
            for(double exponent : exponents) {
              if(exponent > maxExponent) {
                maxExponent = exponent;
              }
            }
            double newValue1 = 0;
            for(double exponent : exponents) {
              newValue1 += Math.exp(exponent - maxExponent);
            }
            double newLogBeta = maxExponent + Math.log(newValue1);
            if (DEBUG) {
              System.out.println("newLogBeta = " + newLogBeta);
            }
            logBetaArray[k - 1][aIndex] = newLogBeta;
          }
        } else {
          ArrayList<Double> exponents = new ArrayList<Double>();
          for (int bIndex = 0; bIndex < dimPOS; bIndex++) {
            // part1 = theta_e(a, xs.get(k))
            double part1 = theta[indexEmission(bIndex, wordIndex.get(xs.get(k)))];
            // part2 = theta_t(a, y0)
            double part2 = theta[indexTransition(POSIndex.get(y0), bIndex)];
            double part3 = logBetaArray[k][bIndex];
            exponents.add(part1 + part2 + part3);
          }

          // use the log sum exp trick to stabilize numerics:
          // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
          double maxExponent = Double.NEGATIVE_INFINITY;
          for(double exponent : exponents) {
            if(exponent > maxExponent) {
              maxExponent = exponent;
            }
          }
          double newValue1 = 0;
          for(double exponent : exponents) {
            newValue1 += Math.exp(exponent - maxExponent);
          }
          double newLogBeta = maxExponent + Math.log(newValue1);
          logBetaArray[k - 1][POSIndex.get(y0)] = newLogBeta;
        }
      }

      logZ = logBetaArray[0][POSIndex.get(y0)];

      if (DEBUG) {
        // we'll calculate it another way to double check
        ArrayList<Double> exponents = new ArrayList<Double>();
        for (int aIndex = 0; aIndex < dimPOS; aIndex++) {
          exponents.add(logAlphaArray[sentence.length()][aIndex]);
        }
        // use the log sum exp trick to stabilize numerics:
        // log(sum_i e^{x_i}) = x_max + log(\sum_i e^(x_i - x_max))
        double maxExponent = Double.NEGATIVE_INFINITY;
        for(double exponent : exponents) {
          if(exponent > maxExponent) {
            maxExponent = exponent;
          }
        }
        double newValue = 0;
        for(double exponent : exponents) {
          newValue += Math.exp(exponent - maxExponent);
        }
        double alternativeLogZ = maxExponent + Math.log(newValue);
        double tolerance = Math.pow(10, -7);
        System.out.println("logZ = " + logZ);
        assert(Math.abs(logZ - alternativeLogZ) < tolerance);
      }
  }
  public void calculateCaches() {
    if(!cacheValid) {
      calculateCachesSentence(trainSentences.get(sentenceCounter));
      cacheValid = true;
    }
  }

  public double[] gradientSentence(Sentence sentence) {
    ArrayList<String> xs = new ArrayList<String>();
        xs.add("");
        xs.addAll(sentence.getAllX());
        ArrayList<String> ys = new ArrayList<String>();
        ys.add(y0);
        ys.addAll(sentence.getAllY());

        if (DEBUG) {
          // check for correctness: sum_a p(y_i = a | x) = 1.
          for(int i = 1; i <= sentence.length(); i++) {
            double theValue = 0;
            for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
              theValue += Math.exp(
                logAlphaArray[i][aIndex] +
                logBetaArray[i][aIndex] -
                logZ
              );
            }
            double tolerance = Math.pow(10, -7);
            assert(Math.abs(theValue - 1.00) < tolerance);
          }
        }

        for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
          for(int bIndex = 0; bIndex < dimPOS; bIndex++) {
            double newValue = 0;
            for(int i = 1; i <= sentence.length(); i++) {
                // add \I[y_{i-1}^{(n)}=a,y_{i}^{(n)}=b]
              if(ys.get(i - 1).equals(POSes.get(aIndex)) && ys.get(i).equals(POSes.get(bIndex))) {
                newValue += 1;
              }

              if(i > 1) {
                newValue -= Math.exp(
                  logAlphaArray[i - 1][aIndex] +
                  theta[indexEmission(bIndex, wordIndex.get(xs.get(i)))] +
                  theta[indexTransition(aIndex, bIndex)] +
                  logBetaArray[i][bIndex] -
                  logZ
                );
              }
            }
            gradient[indexTransition(aIndex, bIndex)] += newValue;
          }
        }

        for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
          for(String c : words) {
            double newValue = 0;
            for(int i = 1; i <= sentence.length(); i++) {
                // add \I[y_{i-1}^{(n)}=a,x_{i}^{(n)}=c]
              if(ys.get(i - 1).equals(POSIndex.get(aIndex)) && xs.get(i).equals(c)) {
                newValue += 1;
              }

              // subtract \I[x_{i}^{(n)}=c]p_{\theta}(y_{i}=a\mid\xx^{(n)})
              if(xs.get(i).equals(c)) {
                newValue -= Math.exp(
                  logAlphaArray[i][aIndex] +
                  logBetaArray[i][aIndex] -
                  logZ
                );
              }
              gradient[indexEmission(aIndex, wordIndex.get(c))] += newValue;
            }
          }
        }
      return gradient;
  }

  public double[] gradient() {
    if (!gradientValid) {
      calculateCaches();
      gradient = gradientSentence(trainSentences.get(sentenceCounter));
    }
    gradientValid = true;
    return gradient;
  }

  public double value() {
    if (!valueValid) {
      calculateCaches();
      value = 0;
      Sentence sentence = trainSentences.get(sentenceCounter);
      ArrayList<String> xs = new ArrayList<String>();
      xs.add("");
      xs.addAll(sentence.getAllX());
      ArrayList<String> ys = new ArrayList<String>();
      ys.add(y0);
      ys.addAll(sentence.getAllY());

      for(int i = 1; i <= sentence.length(); i++) {
        for(int aIndex = 0; aIndex < dimPOS; aIndex++) {
          for(int bIndex = 0; bIndex < dimPOS; bIndex++) {

            if(i > 1 && ys.get(i - 1).equals(POSes.get(aIndex)) && ys.get(i).equals(POSes.get(bIndex))) {
              value += theta[indexTransition(aIndex, bIndex)];
            }
          }

          for(String c : words) {
            if(ys.get(i).equals(POSes.get(aIndex)) && xs.get(i).equals(c)) {
              value += theta[indexEmission(aIndex, wordIndex.get(c))];
            }
          }
        }
      }

      value -= logZ;
    }
    return value;
  }

  public void invalidate() {
    valueValid = gradientValid = cacheValid = false;
  }
}