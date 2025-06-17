import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_community.agent_toolkits.load_tools import load_tools

# Set up API keys
os.environ["SERPAPI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""

# Google model configuration
GOOGLE_MODEL = "gemini-2.0-flash"

class SimpleRAGWithSerpAPI:
    def __init__(self):
        self.search = SerpAPIWrapper()
        # Initialize Google Gemini embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )

        self.llm = ChatGoogleGenerativeAI(
            model=GOOGLE_MODEL,
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )

        # Initialize memory saver
        self.memory = MemorySaver()

        # Initialize vector store (will be created when needed)
        self.vector_store = None

        # Load SerpAPI as a tool
        self.tools = load_tools(["serpapi"])

        # Create agent with memory
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.memory
        )

    def search_and_store(self, query: str):
        """Search web and store results in vector database"""
        # Search using SerpAPI
        print(f"Searching for: {query}")
        search_results = self.search.run(query)

        # Create document from search results
        doc = Document(
            page_content=search_results,
            metadata={"query": query, "source": "serpapi"}
        )

        # Split text for better retrieval
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents([doc])

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

        print(f"Added {len(docs)} documents to vector store")
        return search_results

    def query_with_rag(self, question: str, thread_id: str = "default"):
        """Query using RAG with memory"""
        # First search and store results
        search_results = self.search_and_store(question)

        # Retrieve relevant documents if vector store exists
        context = ""
        if self.vector_store:
            relevant_docs = self.vector_store.similarity_search(question, k=1)
            context = "\n".join([doc.page_content for doc in relevant_docs])

        # Create enhanced prompt with context
        enhanced_question = f"""
            Context from web search: {context}

            Question: {question}

            Please answer based on the context provided above.
            """

        # Use agent with memory
        config = {"configurable": {"thread_id": thread_id}}

        response = self.agent.invoke(
            {"messages": [HumanMessage(content=enhanced_question)]},
            config
        )

        return {
            "answer": response["messages"][-1].content,
            "search_results": search_results,
            "context_used": context[:500] + "..." if len(context) > 500 else context
        }

    def simple_search(self, query: str):
        """Simple web search without RAG"""
        return self.search.run(query)

    def direct_gemini_query(self, question: str):
        """Direct query to Gemini without web search"""
        response = self.llm.invoke([HumanMessage(content=question)])
        return response.content

def main(): # Create the RAG system
    rag_system = SimpleRAGWithSerpAPI()
    # Example 1: Simple search
    print("=== Simple Search ===")
    result = rag_system.simple_search("What is Python programming?")
    print(f"Search Result: {result[:200]}...")

    # Example 2: Direct Gemini query (without web search)
    print("\n=== Direct Gemini Query ===")
    direct_result = rag_system.direct_gemini_query("What is machine learning?")
    print(f"Gemini Response: {direct_result[:200]}...")

    print("\n=== RAG with Memory ===")
    # Example 3: RAG with memory
    questions = [
        "What are the latest AI developments in 2024?", # This will also use memory
    ]

    thread_id = "conversation_1"

    for question in questions:
        print(f"\nQuestion: {question}")
        response = rag_system.query_with_rag(question, thread_id)

        print(f"Answer: {response['answer']}")
        print(f"Context length: {len(response['context_used'])} characters")
        print("-" * 50)

main()
