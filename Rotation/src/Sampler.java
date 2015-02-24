import java.util.Random;

public class Sampler {
	double[] p;
	int numElements;
	Random randomizer;
	
	public Sampler(double[] p, long seed){
		this.p = p;
		numElements = p.length;
		randomizer = new Random(seed);
	}
	
	public int sample() {
		double r = randomizer.nextDouble();		
		double accum_prob = 0.0;
		int the_item = -1;
		for(int i = 0; i < numElements; i++) {
			accum_prob += p[i];
			if(accum_prob >= r) {
				the_item = i;
				break;
			}
		}
		return the_item;
	}
}
