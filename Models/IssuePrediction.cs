using Microsoft.ML.Data;

namespace Step2Issue.Models {
    public class IssuePrediction {
        [ColumnName ("PredictedLabel")]
        public string Area;
    }
}