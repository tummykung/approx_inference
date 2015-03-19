import java.util.ArrayList;



public abstract class Example<X, Y> {
  private ArrayList<X> x;
  private ArrayList<Y> y;

  public Example(ArrayList<X> x, ArrayList<Y> y) {
    this.x = x;
    this.y = y;
  }

  public ArrayList<X> getInput() {
    return x;
  }

  public ArrayList<Y> getOutput() {
    return y;
  }
  
  public void setInput(ArrayList<X> x) {
    this.x = x;
  }

  public void setOutput(ArrayList<Y> y) {
    this.y = y;
  }

  @Override
  public String toString() {
    StringBuilder b = new StringBuilder();
    b.append("input ");
    b.append(x);
    for (int i = 0; i < x.size(); i++) {
      b.append(x.get(i));
      if (i < x.size() - 1)
        b.append(" ");
    }
    b.append("\n");

    b.append("output ");
    for (int i = 0; i < y.size(); i++) {
      b.append(y.get(i));
      if (i < y.size() - 1)
        b.append(" ");
    }
    return b.toString();
  }
}
