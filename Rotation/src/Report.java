import fig.basic.LogInfo;


public class Report {
  public int totalExactMatch;
  public int totalPossibleExactMatch;
  public int totalUnaryMatch;
  public int totalPossibleUnaryMatch;
  public double averageLogLikelihood;
  
  public Report(
      int totalExactMatch,
      int totalPossibleExactMatch,
      int totalUnaryMatch,
      int totalPossibleUnaryMatch,
      double averageLogLikelihood
  ) {
    this.totalExactMatch = totalExactMatch;
    this.totalPossibleExactMatch = totalPossibleExactMatch;
    this.totalUnaryMatch = totalUnaryMatch;
    this.totalPossibleUnaryMatch = totalPossibleUnaryMatch;
    this.averageLogLikelihood = averageLogLikelihood;
  }
  
  public void print(String name) {
    LogInfo.logs(name + ".totalExactMatch:\t" + totalExactMatch);
    LogInfo.logs(name + ".totalPossibleExactMatch:\t" + totalPossibleExactMatch);
    LogInfo.logs(name + ".accuracyExact:\t" + totalExactMatch/(double)totalPossibleExactMatch);
    LogInfo.logs(name + ".totalUnaryMatch:\t" + totalUnaryMatch);
    LogInfo.logs(name + ".totalPossibleUnaryMatch:\t" + totalPossibleUnaryMatch);
    LogInfo.logs(name + ".accuracyUnary:\t" + totalUnaryMatch/(double)totalPossibleUnaryMatch);
    LogInfo.logs(name + ".averageLogLikelihood:\t" + averageLogLikelihood);
  }
}
