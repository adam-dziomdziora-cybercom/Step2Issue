using Microsoft.ML;
using Step2Issue.Models;
using System.Diagnostics;

Console.WriteLine("Hello, World!");

string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "issues_train.tsv");
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "issues_test.tsv");
Directory.CreateDirectory(Path.Combine(Environment.CurrentDirectory, "Models"));
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Models", "model.zip");

var _mlContext = new MLContext();
IDataView _trainingDataView;

_trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(
    _trainDataPath, hasHeader: true);
var debug = _trainingDataView.Preview();
var debugValues = string.Join(", ", debug.RowView[0].Values.Select(v => v.Value.ToString())) ?? "-";
Console.WriteLine(debugValues);

var pipeline = ProcessData(_mlContext);
var trainedModel = BuildAndTrainModel(_mlContext, _trainingDataView, pipeline);
Evaluate(_mlContext, trainedModel, _testDataPath);
SaveModelAsFile(_mlContext, _trainingDataView.Schema, trainedModel, _modelPath);
PredictIssue(_mlContext, _modelPath);


static IEstimator<ITransformer> ProcessData(MLContext mlContext)
{
    var pipeline = mlContext.Transforms.Conversion
        .MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(mlContext.Transforms.Text
            .FeaturizeText(
                inputColumnName: "Title",
                outputColumnName: "TitleFeaturized"))
        .Append(mlContext.Transforms.Text
            .FeaturizeText(
                inputColumnName: "Description",
                outputColumnName: "DescriptionFeaturized"))
        .Append(mlContext.Transforms
            .Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(mlContext);

    return pipeline;
}

static ITransformer BuildAndTrainModel(
    MLContext mlContext,
    IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(mlContext.MulticlassClassification.Trainers
            .SdcaMaximumEntropy("Label", "Features"))
        .Append(mlContext.Transforms.Conversion
            .MapKeyToValue("PredictedLabel"));

    // Train the model fitting to the DataSet
    var stopWatch = new Stopwatch();
    Console.WriteLine($"=============== Training the model  ===============");
    stopWatch.Start();
    var trainedModel = trainingPipeline.Fit(trainingDataView);
    Console.WriteLine($"=============== Finished Training the model Ending time: {stopWatch.ElapsedMilliseconds} ===============");

    var predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(trainedModel);
    var issue = new GitHubIssue()
    {
        Title = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };
    var prediction = predEngine.Predict(issue);
    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

    return trainedModel;
}

static void Evaluate(MLContext mlContext, ITransformer trainedModel, string testDataPath)
{
    var testDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
    var testMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(testDataView));
    Console.WriteLine($"=============== Evaluating to get model's accuracy metrics - Ending time: {DateTime.Now} ===============");
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
}

static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model, string modelPath)
{
    // <SnippetSaveModel> 
    mlContext.Model.Save(model, trainingDataViewSchema, modelPath);
    // </SnippetSaveModel>

    Console.WriteLine("The model is saved to {0}", modelPath);
}

static void PredictIssue(MLContext mlContext, string modelPath)
{
    // <SnippetLoadModel>
    ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
    // </SnippetLoadModel>

    Console.WriteLine($"model Input Schema fields: ${modelInputSchema.Count}");

    // <SnippetAddTestIssue> 
    var singleIssue = new GitHubIssue()
    {
        Title = "Entity Framework crashes",
        Description = "When connecting to the database, EF is crashing"
    };
    // </SnippetAddTestIssue> 

    //Predict label for single hard-coded issue
    // <SnippetCreatePredictionEngine>
    var predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
    // </SnippetCreatePredictionEngine>

    // <SnippetPredictIssue>
    var prediction = predEngine.Predict(singleIssue);
    // </SnippetPredictIssue>

    // <SnippetDisplayResults>
    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
    // </SnippetDisplayResults>

}
