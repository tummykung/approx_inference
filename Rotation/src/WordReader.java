import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

public class WordReader{

    public static ArrayList<String> read(String filename) throws Exception {
        StringTokenizer st;
        BufferedReader File = new BufferedReader(new FileReader(filename));

        ArrayList<String> words = new ArrayList<String>();

        String line = File.readLine(); // Read first line.

        while (line != null){
            st = new StringTokenizer(line);
            while(st.hasMoreElements()){
                words.add(st.nextElement().toString());
            }
            line = File.readLine(); // Read the next line
        }
        File.close();
        return words;
    }

    public static void main(String[] args) throws Exception {
      List<String> words = read("data/just_words.txt");
      for (String string : words) {
        System.out.println(string);
      }
    }
}