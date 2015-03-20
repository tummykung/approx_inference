import java.util.ArrayList;

public class AlignmentExample extends IndirectSupervisionExample {
  
  public int[] denotation;
  
  public AlignmentExample(int[] x, int[] y, int[] z) {
    super(x, y, z);
    this.denotation = denotation();
  }

  public AlignmentExample(int[] x, int[] y, ArrayList<String> z_orig) throws Exception {
    super(x, y, convertZToInt(z_orig));
    this.denotation = denotation();
  }
  public AlignmentExample(ArrayList<Character> x, 
                          ArrayList<Character> y, 
                          ArrayList<String> z_orig) throws Exception{
    super(convertXToInt(x), convertYToInt(y), convertZToInt(z_orig));
    this.denotation = denotation();
  }

  public static int[] convertZToInt(ArrayList<String> lst) throws Exception{
    int[] ret = new int[lst.size()];
    
    for(int i = 0; i < lst.size(); i++){
      String str = lst.get(i);
      if(str.charAt(0) == 'O'){
        ret[i] = 2 * Main.alphabetSize;
      } else if(str.charAt(0) == 'B'){
        ret[i] = (int)(str.charAt(2) - 'a');
      } else if(str.charAt(0) == 'I'){
        ret[i] = Main.alphabetSize + (int)(str.charAt(2) - 'a');
      } else {
        throw new Exception("We shouldn't be here");
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
  
  public static int[] denotation(int[] z) {
 // only output things that start with 'B', which is characterized
    // the index being in between (inclusive) 0 and alphabetSize
    ArrayList<Integer> to_ret = new ArrayList<Integer>();
    for(int i = 0; i < z.length; i++) {
      if(z[i] < Main.alphabetSize) {
        to_ret.add(z[i]);
      }
    }
    int[] ret = new int[to_ret.size()];
    for(int i = 0; i < to_ret.size(); i++) {
      ret[i] = to_ret.get(i);
    }
    return ret;
  }
  
  private int[] denotation() {
    return denotation(z);
  }
  
  public static String convertIntToString(int i) throws Exception {
    if (i > 2 * Main.alphabetSize || i < 0) {
      throw new Exception("Out of range, try again");
    } else if (i == 2 * Main.alphabetSize) {
      return "O";
    } else if (i < Main.alphabetSize) {
      return "B-" + (char)('a' + i); 
    } else {
      return "I-" + (char)('a' + i - Main.alphabetSize);
    }
  }
  
  public static String toStringHumanReadable(int[] characters, String name) throws Exception {
    StringBuilder b = new StringBuilder();
    b.append(name + " ");
    for (int i = 0; i < characters.length; i++) {
      b.append((char)('a' + characters[i]));
      if (i < characters.length - 1)
        b.append(" ");
    }
    return b.toString();
  }
  public String toStringHumanReadable() throws Exception {
    StringBuilder b = new StringBuilder();
    b.append(toStringHumanReadable(x, "input"));
    b.append("\n");
    b.append("denotation ");
    for (int i = 0; i < denotation.length; i++) {
      b.append((char)('a' + denotation[i]));
      if (i < denotation.length - 1)
        b.append(" ");
    }
    b.append("\n");
    b.append(toStringHumanReadable(z, "latent"));
    b.append("\n");
    b.append(toStringHumanReadable(y, "output"));
    return b.toString();
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

    b.append("latent ");
    for (int i = 0; i < z.length; i++) {
      b.append(z[i]);
      if (i < z.length - 1)
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
