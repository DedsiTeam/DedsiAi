#pragma warning disable SKEXP0070
#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0050
#pragma warning disable CS0618

using System.Text;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.Memory;

var endpoint = new Uri("http://localhost:11434");
var modelId = "deepseek-r1:8b";
var textEmbeddingModelId = "mxbai-embed-large:latest";
const string collectionName = "LatestNews";

var kernelBuild = Kernel.CreateBuilder()
    .AddOllamaChatCompletion(modelId, endpoint)
    .AddOllamaTextEmbeddingGeneration(textEmbeddingModelId, endpoint)
    .Build();

// 创建文本向量生成服务
var embeddingGenerator = kernelBuild.GetRequiredService<ITextEmbeddingGenerationService>();

// 创建文本向量生成服务
var memory = new MemoryBuilder()
    .WithMemoryStore(new VolatileMemoryStore())
    .WithTextEmbeddingGeneration(embeddingGenerator)
    .Build();

var paragraphs = new []
{
    "合同Id: 1234567890, 合同编号：1234567890, 合同名称：测试合同, 合同金额：10000元, 合同日期：2021-01-01, 合同甲方：王小龙，合同乙方：李小龙;",
    "合同Id: 1234567891, 合同编号：1234567891, 合同名称：测试合同, 合同金额：2000000元, 合同日期：2021-01-01, 合同甲方：王小龙，合同乙方：王明;",
    "合同Id: 1234567892, 合同编号：1234567892, 合同名称：测试合同, 合同金额：3000元, 合同日期：2021-01-01, 合同甲方：王小龙，合同乙方：张丽华;",
    "合同Id: 1234567893, 合同编号：1234567893, 合同名称：测试合同, 合同金额：10000.991元, 合同日期：2021-01-01, 合同甲方：王小龙，合同乙方：张振;",
    "合同Id: 1234567894, 合同编号：1234567894, 合同名称：测试合同, 合同金额：2000元, 合同日期：2021-01-01, 合同甲方：王小龙，合同乙方：王位;",
};

// 将各个段落进行量化并保存到向量数据库
for (var i = 0; i < paragraphs.Length; i++)
{
    await memory.SaveInformationAsync(collectionName, paragraphs[i], Guid.NewGuid().ToString());
}

var chatCompletionService = kernelBuild.GetRequiredService<IChatCompletionService>();
var chat = new ChatHistory("你是一个AI助手，帮助人们查找信息和回答问题");
StringBuilder additionalInfo = new();
StringBuilder chatResponse = new();

while (true)
{
    additionalInfo.Clear();
    Console.WriteLine("请输入问题>> ");
    var question = Console.ReadLine();
    
    // 从向量数据库中找到跟提问最为相近的3条信息，将其添加到对话历史中
    await foreach (var hit in memory.SearchAsync(collectionName, question, limit: paragraphs.Length))
    {
        additionalInfo.AppendLine(hit.Metadata.Text);
    }
    
    var contextLinesToRemove = -1;
    if (additionalInfo.Length != 0)
    {
        contextLinesToRemove = chat.Count;
        additionalInfo.Insert(0, "以下是一些附加信息：");
        chat.AddUserMessage(additionalInfo.ToString());
    }
    chat.AddUserMessage(question);

    chatResponse.Clear();
    await foreach (var message in chatCompletionService.GetStreamingChatMessageContentsAsync(chat))
    {
        Console.Write(message);
        // 将输出内容添加到临时变量中
        chatResponse.Append(message.Content);
    }
    // 在进入下一次问答之前，将当前回答结果添加到对话历史中，为大语言模型提供问答上下文
    chat.AddAssistantMessage(chatResponse.ToString());
    // 将当次问题相关的内容从对话历史中移除
    if (contextLinesToRemove >= 0)
    {
        chat.RemoveAt(contextLinesToRemove);
    }
}