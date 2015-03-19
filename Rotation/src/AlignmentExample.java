public class AlignmentExample extends IndirectSupervisionExample<Character, Character, String> {
  public AlignmentExample(ArrayList<Character> x, 
                          ArrayList<Character> y, 
                          ArrayList<String> z_orig){
    super(x, y, z_orig);
  }

  @Override
  public int[] convertToInt(ArrayList<String> lst){
    int[] ret = new int[lst.size()];
    int alphabetSize = 26;
    for(int i = 0; i < lst.size(); i++){
      String str = lst.get(i);
      if(str.charAt(0) == '0'){
        ret[i] = 2 * alphabetSize;
      } else if(str.charAt(0) == 'B'){
        ret[i] = (int)(str.charAt(0) - 'a');
      } else if(str.charAt(0) == 'I'){
        ret[i] = alphabetSize + (int)(str.charAt(0) - 'a');
      }
    }
    return ret;
  }

}
