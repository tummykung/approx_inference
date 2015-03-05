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
  
  public void setInput(int[] x) {
    this.x = x;
  }

  public void setOutput(int[] y) {
    this.y = y;
  }

  @Override
  public String toString() {
    StringBuilder b = new StringBuilder();
    b.append("input ");
    for (int i = 0; i < x.length; i++) {
      b.append(x[i]);
      if (i < x.length - 1)
        b.append(" ");
    }
    b.append("\n");

    b.append("output ");
    for (int i = 0; i < y.length; i++) {
      b.append(y[i]);
      if (i < y.length - 1)
        b.append(" ");
    }
    return b.toString();
  }
}
