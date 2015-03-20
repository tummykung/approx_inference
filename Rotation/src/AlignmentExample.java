import java.util.ArrayList;

public class AlignmentExample extends IndirectSupervisionExample {
  
  public AlignmentExample(int[] x, int[] y, int[] z) {
    super(x, y, z);
  }

  public AlignmentExample(int[] x, int[] y, ArrayList<String> z_orig) {
    super(x, y, convertZToInt(z_orig));
    
  }
  public AlignmentExample(ArrayList<Character> x, 
                          ArrayList<Character> y, 
                          ArrayList<String> z_orig){
    super(convertXToInt(x), convertYToInt(y), convertZToInt(z_orig));
  }

  public static int[] convertZToInt(ArrayList<String> lst){
    int[] ret = new int[lst.size()];
    
    for(int i = 0; i < lst.size(); i++){
      String str = lst.get(i);
      if(str.charAt(0) == '0'){
        ret[i] = 2 * Main.alphabetSize;
      } else if(str.charAt(0) == 'B'){
        ret[i] = (int)(str.charAt(0) - 'a');
      } else if(str.charAt(0) == 'I'){
        ret[i] = Main.alphabetSize + (int)(str.charAt(0) - 'a');
      }
    }
    return ret;
  }

  public static int[] convertXToInt(ArrayList<Character> lst){
    int[] ret = new int[lst.size()];
    for(int i = 0; i < lst.size(); i++){
      ret[i] = lst.get(i) - 'a';
    }
    return ret;
  }

  public static int[] convertYToInt(ArrayList<Character> lst){
    return convertXToInt(lst);
  }
}
