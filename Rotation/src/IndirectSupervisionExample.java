public class IndirectSupervisionExample extends Example{
  protected int[] z;

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
