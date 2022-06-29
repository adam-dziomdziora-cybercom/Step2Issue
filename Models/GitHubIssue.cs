using Microsoft.ML.Data;

namespace Step2Issue.Models;

public class GitHubIssue
{
    [LoadColumn(0)]
    public string ID { get; set; } = string.Empty;

    [LoadColumn(1)]
    public string Area { get; set; } = string.Empty;

    [LoadColumn(2)]
    public string Title { get; set; } = string.Empty;

    [LoadColumn(3)]
    public string Description { get; set; } = string.Empty;
}
