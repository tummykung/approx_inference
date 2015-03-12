import java.util.Arrays;
import java.util.Random;


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
  
  /**
   * Returns a pseudo-random number between min and max, inclusive.
   * The difference between min and max can be at most
   * <code>Integer.MAX_VALUE - 1</code>.
   *
   * @param min Minimum value
   * @param max Maximum value.  Must be greater than min.
   * @return Integer between min and max, inclusive.
   * @see java.util.Random#nextInt(int)
   * http://stackoverflow.com/questions/363681/generating-random-integers-in-a-range-with-java
   */
  public static int randInt(int min, int max) {

      // NOTE: Usually this should be a field rather than a method
      // variable so that it is not re-seeded every call.
      Random rand = new Random();

      // nextInt is normally exclusive of the top value,
      // so add 1 to make it inclusive
      int randomNum = rand.nextInt((max - min) + 1) + min;

      return randomNum;
  }
}
