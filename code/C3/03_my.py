from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 参考 02_langchain_faiss.py，实现对 03_llamaindex_vector.py 持久化索引的加载与检索

# 1. 配置与创建索引时相同的嵌入模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cpu"
)

# 2. 从本地加载已持久化的 LlamaIndex 索引
persist_path = "./llamaindex_index_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_path)
index = load_index_from_storage(storage_context)

print(f"LlamaIndex index has been loaded from {persist_path}")

# 3. 执行相似检索
query = "LlamaIndex是做什么的？"
retriever = index.as_retriever(similarity_top_k=1)
results = retriever.retrieve(query)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for node in results:
    print(f"- {node.text}")
