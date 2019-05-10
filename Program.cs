using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Step2Issue.Models;

namespace Step2Issue {
    class Program {
        private static string _trainDataPath => Path.Combine (Environment.CurrentDirectory, "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine (Environment.CurrentDirectory, "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine (Environment.CurrentDirectory, "Models", "model.zip");

        private static MLContext _mlContext = new MLContext ();
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;
        static void Main (string[] args) {
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue> (_trainDataPath, hasHeader : true);

        }
        public static IEstimator<ITransformer> ProcessData () {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey (inputColumnName: "Area", outputColumnName: "Label")
                .Append (_mlContext.Transforms.Text
                    .FeaturizeText (
                        inputColumnName: "Title",
                        outputColumnName: "TitleFeaturized"))
                .Append (_mlContext.Transforms.Text
                    .FeaturizeText (
                        inputColumnName: "Description",
                        outputColumnName: "DescriptionFeaturized"))
                .AppendCacheCheckpoint (_mlContext);

            return pipeline;
        }

    }
}