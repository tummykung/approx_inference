import java.util.ArrayList;


public class LinearChainCRF {
	public int num_layers;
	public int arities = 3;
	public int ranges;
	public int dim_theta;
	public int potentials;
	public boolean learning_verbose = false;
	public boolean state_verbose = false;
	public boolean log_likelihood_verbose = true;
	public boolean prediction_verbose = false;
	public boolean sanity_check = false;
	public boolean fully_supervised = false;
	public ArrayList<Integer> m;
	public int rangeX;
	public int sentenceLength;
	public double xi;
	public long seed;
	public static Sampler sampler;
	
	public LinearChainCRF(long seed) {
		this.seed = seed;
		num_layers = 3;
		m = new ArrayList<Integer>();
		m.add(3);
		m.add(3);
		m.add(9);
		rangeX = 5;
		sentenceLength = 3;
		xi = 10.0;
		int numElements = 10;
		double[] p = new double[numElements]; 
		sampler = new Sampler(p, seed);
	}
	
	public void generate_data(double[] parameters) {	
	}
	
	public int phi0(int[] y, int[] x) {
		return indicator(y[0] == x[0]);
	}
	public int phi1(int[] y, int[] x) {
		return indicator(y[1] == x[1]);
	}
	public int phi2(int[] y, int[] x) {
		return indicator(y[2] == x[2]);
	}
	public double calculateZ(int[] x, double[] parameters){
		return 1;
	}
	
	public double p(int[] z, int[] x, double[] parameters) {
		double exponent = 0.0;
		for(int i = 0 ; i < arities; i++) {
			if(z[i] == x[i]) {
				exponent += parameters[i];
			}
		}
		return Math.exp(exponent)/calculateZ(x, parameters);
	}
	
	public int indicator(boolean bool) {
		if (bool) {
			return 1;
		} else {
			return 0;
		}
	}
}
