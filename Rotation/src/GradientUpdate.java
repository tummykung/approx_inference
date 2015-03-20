
public class GradientUpdate {

  public static void update(Params thetaHat, Params thetaHatAverage, Params gradient, Params st, int counter, double eta0) {
    double eta = 0.0;
    if (Main.gradientDescentType == Main.DECREASING_STEP_SIZES) {
      eta = Main.eta0/Math.sqrt(counter);
    } else if (Main.gradientDescentType == Main.CONSTANT_STEP_SIZES){
      eta = Main.eta0;
    }
    // if AdaGrad we'll do at the level of update itself

    if (Main.extraVerbose) {
      gradient.print("gradient");
      thetaHat.print("thetaHat");
      thetaHatAverage.print("thetaHatAverage");
    }

    for (int a = 0; a <= Main.rangeZ; a++)
      for (int b = 0; b <= Main.rangeZ; b++) {
        if (Main.gradientDescentType == Main.ADAGRAD){
          st.transitions[a][b] += gradient.transitions[a][b] * gradient.transitions[a][b];
          thetaHat.transitions[a][b] += Main.eta0/Math.sqrt(st.transitions[a][b]) * gradient.transitions[a][b];
        } else
          thetaHat.transitions[a][b] += eta * gradient.transitions[a][b];
        thetaHatAverage.transitions[a][b] =
            (counter - 1)/(double)counter*thetaHatAverage.transitions[a][b] +
            1/(double)counter*thetaHat.transitions[a][b];
      }
    for (int a = 0; a <= Main.rangeZ; a++)
      for (int c = 0; c <= Main.rangeX; c++) {
        if (Main.gradientDescentType == Main.ADAGRAD){
          st.emissions[a][c] += gradient.emissions[a][c] * gradient.emissions[a][c];
          thetaHat.emissions[a][c] += Main.eta0/Math.sqrt(st.emissions[a][c]) * gradient.emissions[a][c];
        } else
          thetaHat.emissions[a][c] += eta * gradient.emissions[a][c];
        thetaHatAverage.emissions[a][c] =
            (counter - 1)/(double)counter*thetaHatAverage.emissions[a][c] +
            1/(double)counter*thetaHat.emissions[a][c];
      }
  }
}
