from langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import requests

llm = ChatOpenAI(model="gpt-4o-mini")

def sample1():
    # Load, chunk and index the contents of the blog.

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    url_list = ['2023-06-23-agent/', '2023-03-15-prompt-engineering/', '2023-10-25-adv-attack-llm/']
    docs = {}
    for paths in url_list:
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/"+paths,),
            session=session,  # 여기에 session 전달
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )

        docs[paths] = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n\n\n"],chunk_size=2000, chunk_overlap=200, length_function=len,
                                                    is_separator_regex=False)
        splits = text_splitter.split_documents(docs[paths])

    print(splits[0])

def sample2():
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    # print(prompt)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(rag_chain.invoke("What is Task Decomposition?"))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    loader = WebBaseLoader(
        web_paths=urls
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n", "\n\n", "\n\n\n"],chunk_size=2000, chunk_overlap=200, length_function=len,
                                                    is_separator_regex=False)
    splits = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template="You are a grader assessing relevance of a retrived document to a user query. Determine the relevance between query and document. The output should be formatted as a JSON instance that has key name relevance and value is yes or no.\nquery: {query}\ndocument: {document}\n",
        input_variables=["query", "document"]
    )

    chain = (
        {"document": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    
    for x in chain.invoke("llm blog"):
        print(x)


main()
