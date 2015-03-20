import java.util.ArrayList;


abstract class IndirectSupervisionExample<X, Y, Z> extends Example{

  private ArrayList<Z> z_orig;
  private int[] z;
  private ArrayList<Y> y_orig;
  private int[] y;
  private ArrayList<X> x_orig;
  private int[] x;

  public IndirectSupervisionExample(int[] x, int[] y, int[] z) {
    super(x, y);
    this.z = z;
  }

  public int[] getLatent() {
    return z;
  }
  
  public void setLatent(int[] z) {
    this.z = z;
  }
}
