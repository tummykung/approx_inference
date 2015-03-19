import java.util.ArrayList;


public class IndirectSupervisionExample<X, Y, Z> extends Example<X, Y>{

  private ArrayList<Z> z;

  public IndirectSupervisionExample(ArrayList<X> x, ArrayList<Y> y, ArrayList<Z> z) {
    super(x, y);
    this.z = z;
  }

  public ArrayList<Z> getLatent() {
    return z;
  }
  
  public void setLatent(ArrayList<Z> z) {
    this.z = z;
  }
}
