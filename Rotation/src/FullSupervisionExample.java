import java.util.ArrayList;


public class FullSupervisionExample extends Example {

  private int[] y;
  private int[] x;
  
  public FullSupervisionExample(int[] x, int[] y) {
    super(x, y);
  }
  
  public int[] getX() {
    return x;
  }

  public int[] getY() {
    return y;
  }

  public int[] convertXToInt(ArrayList<Integer> lst){
    int[] ret = new int[lst.size()];
    for(int i = 0; i < lst.size(); i++){
      ret[i] = lst.get(i);
    }
    return ret;
  }

  public int[] convertYToInt(ArrayList<Integer> lst){
    return convertXToInt(lst);
  }
}
