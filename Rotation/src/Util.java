import java.util.Arrays;


public class Util {
  // not really necessary; can be replaced by Arrays.deepToString
  public static String intArrayPrettyPrint(int[] array) {
    String output = "";
    output += "[";
    for (int i = 0; i < array.length - 1; i++) {
      int j = array[i];
      output += j + ", ";
    }
    // the last one: no trailing command and space
    int j = array[array.length - 1];
    output += j;
    output += "]";
    return output;
  }
  
  public static int indicator(boolean bool) {
    if (bool) {
      return 1;
    } else {
      return 0;
    }
  }
}
