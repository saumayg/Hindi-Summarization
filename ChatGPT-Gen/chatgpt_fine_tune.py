from openai import OpenAI
client = OpenAI(api_key="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")

client.files.create(
    file=open("../Datasets/Hindi_summarization/XLSum/ChatGPT/chatgpt_train_hindi_9.jsonl", "rb"),
    purpose="fine-tune"
)

client.files.create(
    file=open("../Datasets/Hindi_summarization/XLSum/ChatGPT/chatgpt_test_hindi_9.jsonl", "rb"),
    purpose="fine-tune"
)

# client.fine_tuning.jobs.create(
#     training_file="file-ZOag6hW4J7FDElSLkOsBB2D8",
#     model="gpt-3.5-turbo-1106"
# )