import os
from langchain_openai import ChatOpenAI
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import logging
import re
import requests

logging.basicConfig(level=logging.INFO)


def extract_bvid(video_url: str) -> str | None:
    match = re.search(r"(BV[0-9A-Za-z]+)", video_url)
    return match.group(1) if match else None


def load_bilibili_documents(video_urls: list[str]) -> list[Document]:
    """通过 B 站公开视频接口加载视频信息，避免页面响应使用 br 编码导致解码失败。"""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
        }
    )

    documents: list[Document] = []
    for video_url in video_urls:
        bvid = extract_bvid(video_url)
        if not bvid:
            logging.warning("无法从链接中提取 BV 号: %s", video_url)
            continue

        try:
            response = session.get(
                "https://api.bilibili.com/x/web-interface/view",
                params={"bvid": bvid},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            logging.warning("请求视频信息失败 %s: %s", bvid, exc)
            continue
        except ValueError as exc:
            logging.warning("解析视频信息失败 %s: %s", bvid, exc)
            continue

        if payload.get("code") != 0 or not payload.get("data"):
            logging.warning("B 站接口返回异常 %s: %s", bvid, payload)
            continue

        original = payload["data"]
        metadata = {
            "title": original.get("title", "未知标题"),
            "author": original.get("owner", {}).get("name", "未知作者"),
            "source": original.get("bvid", bvid),
            "view_count": original.get("stat", {}).get("view", 0),
            "length": original.get("duration", 0),
        }
        page_content = "\n".join(
            [
                f"标题：{metadata['title']}",
                f"作者：{metadata['author']}",
                f"简介：{original.get('desc', '')}".strip(),
            ]
        ).strip()
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents

# 1. 初始化视频数据
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

bili = []
try:
    bili = load_bilibili_documents(video_urls)
except Exception as e:
    print(f"加载BiliBili视频失败: {str(e)}")

if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'}
)
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 创建自查询检索器
llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0,
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    metadata_field_info=metadata_field_info,
    enable_limit=True,
    verbose=True
)

# 5. 执行查询示例
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    results = retriever.invoke(query)
    if results:
        for doc in results:
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        print("未找到匹配的视频")
