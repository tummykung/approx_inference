import java.util.Arrays;


public class Example {
  private int[] x;
  private int[] y;

  public Example(int[] x, int[] y) {
    this.x = x;
    this.y = y;
  }

  public int[] getInput() {
    return x;
  }

  public int[] getOutput() {
    return y;
  }

  @Override
  public String toString() {
    return "Input: " + Arrays.toString(x) + "\tOutput:\t" + Arrays.toString(y) + "\n";
  }
}
