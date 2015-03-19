import java.util.ArrayList;


abstract class IndirectSupervisionExample<X, Y, Z> extends Example<X, Y>{

  private ArrayList<Z> z_orig;
  private int[] z;

  public IndirectSupervisionExample(ArrayList<X> x, ArrayList<Y> y, ArrayList<Z> z_orig) {
    super(x, y);
    this.z_orig = z_orig;
    this.z = convertToInt(z_orig);
  }

  abstract int[] convertToInt(ArrayList<Z> lst);

  public int[] getLatent() {
    return z;
  }
  
  public void setLatent(ArrayList<Z> z) {
    this.z = z;
  }
}
