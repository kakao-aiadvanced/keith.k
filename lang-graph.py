from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tavily import TavilyClient


llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
tavily = TavilyClient(api_key='tvly-dev-xxxxxxxxxx')
retriever = None
question_router = None
retrieval_grader = None
rag_chain = None
hallucination_grader = None
answer_grader = None


def initialize():
    global retriever
    global question_router
    global retrieval_grader
    global rag_chain
    global hallucination_grader
    global answer_grader

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()
    logger.info(f"Indexed {len(doc_splits)} documents into vectorstore.")

    ### Router
    system = """You are an expert at routing a user question to a vectorstore or web search.
    Use the vectorstore for questions on LLM agents, prompt, prompt engineering, and adversarial attacks.
    You do not need to be stringent with the keywords in the question related to these topics.
    Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
    Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}"),
        ]
    )

    question_router = prompt | llm | JsonOutputParser()
    logger.info("Initialized question router.")


    ### Retrieval Grader
    system = """You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n document: {document} "),
        ]
    )

    retrieval_grader = prompt | llm | JsonOutputParser()
    logger.info("Initialized retrieval grader.")


    ### Generate
    system = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        Append the sources of the information you used to answer the question at the end of your answer."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n context: {context} "),
        ]
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()


    ### Hallucination Grader
    system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    hallucination_grader = prompt | llm | JsonOutputParser()

    ### Grader
    # Prompt
    system = """You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "question: {question}\n\n answer: {generation} "),
        ]
    )

    answer_grader = prompt | llm | JsonOutputParser()

def debug():
    question = "What is prompt?"
    docs = retriever.get_relevant_documents(question)
    logger.debug(question_router.invoke({"question": question}))
    logger.debug(docs[0].metadata)
    logger.debug(docs[0].page_content)

    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content
    result = retrieval_grader.invoke({"question": question, "document": doc_txt})
    logger.debug(result)
    logger.debug(doc_txt)

    generation = rag_chain.invoke({"context": docs, "question": question})
    logger.debug(generation)

    logger.debug(hallucination_grader.invoke({"documents": docs, "generation": generation}))

    logger.debug(answer_grader.invoke({"question": question, "generation": generation}))



### State


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    relevance_counter: int
    generation_counter: int


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    logger.debug(question)
    logger.debug(documents)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation_counter = state.get("generation_counter", 0) + 1

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "generation_counter": generation_counter}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    relevance_counter = state.get("relevance_counter", 0) + 1

    # Score each doc
    filtered_docs = []
    
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "relevance_counter": relevance_counter}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = None
    if "documents" in state:
      documents = state["documents"]

    # Web search
    docs = tavily.search(query=question)['results']

    for d in docs:
        logger.debug(f"Web search result: {d['url']} {d['title']}")

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}


### Edges


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        if state.get("relevance_counter", 0) > 1:
            print(
                "---DECISION: FAIL, NOT RELEVANT---"
            )
            return "fail"
        else:
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
        

        # print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # # Check question-answering
        # print("---GRADE GENERATION vs QUESTION---")
        # score = answer_grader.invoke({"question": question, "generation": generation})
        # grade = score["score"]
        # if grade == "yes":
        #     print("---DECISION: GENERATION ADDRESSES QUESTION---")
        #     return "useful"
        # else:
        #     print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        #     return "not useful"
    else:
        if state.get("generation_counter", 0) > 1:
            print("---DECISION: FAIL, HALLUCINATION---")
            return "fail"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

def fail_relevance(state):
    """
    Fail node, if we cannot generate an answer

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    
    return {"generation": {"failed": "not relevant"}}

def fail_hallucination(state):
    """
    Fail node, if we cannot generate an answer

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    
    return {"generation": {"failed": "hallucination"}}

def run_graph():
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("fail_relevance", fail_relevance)  # fail
    workflow.add_node("fail_hallucination", fail_hallucination)  # fail


    # Build graph
    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
            "fail": "fail_relevance"
        },
    )
    workflow.add_edge("websearch", "grade_documents")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "fail": "fail_hallucination",
            "not useful": "websearch",
        },
    )
    workflow.add_edge("fail_relevance", END)
    workflow.add_edge("fail_hallucination", END)

    # Compile
    app = workflow.compile()


    # Test
    inputs = {"question": "What is prompt?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])



if __name__ == '__main__':
    logger = logging.getLogger("my")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    
    logger.info("==== Starting Lang Graph...")

    initialize()

    # debug()
    
    run_graph()

    logger.info("==== Lang Graph completed successfully.")
