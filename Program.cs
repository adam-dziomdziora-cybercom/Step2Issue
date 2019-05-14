using System;
using System.Diagnostics;
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
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue> (
                _trainDataPath, hasHeader : true);
            var debug = _trainingDataView.Preview ();

            var pipeline = ProcessData ();
            var trainingPipeline = BuildAndTrainModel (_trainingDataView, pipeline);
            Evaluate (_trainingDataView.Schema);

        }
        public static IEstimator<ITransformer> ProcessData () {
            var pipeline = _mlContext.Transforms.Conversion
                .MapValueToKey (inputColumnName: "Area", outputColumnName: "Label")
                .Append (_mlContext.Transforms.Text
                    .FeaturizeText (
                        inputColumnName: "Title",
                        outputColumnName: "TitleFeaturized"))
                .Append (_mlContext.Transforms.Text
                    .FeaturizeText (
                        inputColumnName: "Description",
                        outputColumnName: "DescriptionFeaturized"))
                .Append (_mlContext.Transforms
                    .Concatenate ("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint (_mlContext);

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel (
            IDataView trainingDataView, IEstimator<ITransformer> pipeline) {
            var trainingPipeline = pipeline
                .Append (_mlContext.MulticlassClassification.Trainers
                    .SdcaMaximumEntropy ("Label", "Features"))
                .Append (_mlContext.Transforms.Conversion
                    .MapKeyToValue ("PredictedLabel"));

            // Train the model fitting to the DataSet
            Stopwatch stopWatch = new Stopwatch ();
            Console.WriteLine ($"=============== Training the model  ===============");
            stopWatch.Start ();
            _trainedModel = trainingPipeline.Fit (trainingDataView);
            Console.WriteLine ($"=============== Finished Training the model Ending time: {stopWatch.ElapsedMilliseconds} ===============");

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction> (_trainedModel);
            GitHubIssue issue = new GitHubIssue () {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            var prediction = _predEngine.Predict (issue);
            Console.WriteLine ($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

            return trainingPipeline;
        }

        public static void Evaluate (DataViewSchema trainingDataViewSchema) {
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue> (_testDataPath, hasHeader : true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate (_trainedModel.Transform (testDataView));
            Console.WriteLine ($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now.ToString()} ===============");
            Console.WriteLine ($"*************************************************************************************************************");
            Console.WriteLine ($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine ($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine ($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine ($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine ($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine ($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine ($"*************************************************************************************************************");
            SaveModelAsFile (_mlContext, trainingDataViewSchema, _trainedModel);

        }

        private static void SaveModelAsFile (MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model) {
            // <SnippetSaveModel> 
            mlContext.Model.Save (model, trainingDataViewSchema, _modelPath);
            // </SnippetSaveModel>

            Console.WriteLine ("The model is saved to {0}", _modelPath);
        }

    }
}