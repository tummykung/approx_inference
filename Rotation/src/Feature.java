

public interface Feature {
  
  public static abstract class NodeFeature implements Feature {
  }
  
  public static abstract class Indicator implements Feature {
  }
  
  public static class ExactMatch extends Indicator {
    
  }
}