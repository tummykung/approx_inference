import java.util.ArrayList;

public class Sentence {
	public ArrayList<Token> tokens;
	public ArrayList<String> Xs;
	public ArrayList<String> Ys;
	public ArrayList<String> Zs;

	public Sentence() {
		tokens = new ArrayList<Token>();
		Xs = new ArrayList<String>();
		Ys = new ArrayList<String>();
		Zs = new ArrayList<String>();
	}

	public Sentence(ArrayList<Token> tokens) { this.tokens = tokens; }

	public void addToken(Token token) {
		tokens.add(token);
		Xs.add(token.x);
		Ys.add(token.y);
		Zs.add(token.z);
	}

	public Token getToken(int i) { return tokens.get(i); }

	public ArrayList<Token> getAllTokens(Token token) { return tokens; }

	public int length() { return tokens.size(); }

	public ArrayList<String> getAllX() { return Xs; }

	public ArrayList<String> getAllY() { return Ys; }

	public ArrayList<String> getAllZ() { return Zs; }

	@Override
	public String toString() {
		String returned_string = "";
		for (Token token : tokens) {
			returned_string += token.x + " ";
		}
		return returned_string;
	}
}