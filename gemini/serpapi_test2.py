import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

        # Initialize vector store (will be created when needed)
        self.vector_store = None

        # Initialize conversation memory
        self.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    def search_and_store(self, query: str):
        """Search web and store results in vector database"""
        print(f"Searching for: {query}")
        search_results = self.search.run(query)
        print(f"Search results length: {len(search_results)} characters")

        # Create document from search results
        doc = Document(
            page_content=search_results,
            metadata={"query": query, "source": "serpapi"}
        )

        # Split text for better retrieval
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents([doc])

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)

        print(f"Added {len(docs)} documents to vector store")
        return search_results

    def query_with_rag(self, question: str):
        """Query using RAG with direct LLM call (not agent)"""
        # First search and store results
        search_results = self.search_and_store(question)

        # Retrieve relevant documents
        context = ""
        if self.vector_store:
            relevant_docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            print(f"Retrieved context length: {len(context)} characters")

        # Create messages for direct LLM call
        system_message = SystemMessage(content="""
        You are a helpful AI assistant. Use the provided context from web search results to answer the user's question accurately. 
        If the context doesn't contain relevant information, say so clearly.
        Always base your answers on the provided context.
        """)

        human_message = HumanMessage(content=f"""
        Context from web search:
        {context}

        Question: {question}

        Please provide a comprehensive answer based on the context above.
        """)

        # Get response directly from LLM
        messages = [system_message, human_message]
        response = self.llm.invoke(messages)

        # Save to conversation memory
        self.conversation_memory.save_context(
            {"input": question},
            {"answer": response.content}
        )

        return {
            "answer": response.content,
            "search_results": search_results,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
            "context_length": len(context)
        }

    def query_with_memory(self, question: str):
        """Query with conversation history"""
        # Get previous conversation
        chat_history = self.conversation_memory.chat_memory.messages

        # First search and store results
        search_results = self.search_and_store(question)

        # Retrieve relevant documents
        context = ""
        if self.vector_store:
            relevant_docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Include chat history in the prompt
        history_text = ""
        if chat_history:
            history_text = "Previous conversation:\n"
            for msg in chat_history[-4:]:  # Last 4 messages
                role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                history_text += f"{role}: {msg.content[:200]}...\n"

        system_message = SystemMessage(content="""
        You are a helpful AI assistant with access to current web search results and conversation history.
        Use both the provided context and conversation history to give comprehensive answers.
        """)

        human_message = HumanMessage(content=f"""
        {history_text}

        Current context from web search:
        {context}

        Current question: {question}

        Please provide a comprehensive answer considering both the current context and our conversation history.
        """)

        response = self.llm.invoke([system_message, human_message])

        # Save to memory
        self.conversation_memory.save_context(
            {"input": question},
            {"answer": response.content}
        )

        return {
            "answer": response.content,
            "search_results": search_results,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
            "has_history": len(chat_history) > 0
        }

    def simple_search(self, query: str):
        """Simple web search without RAG"""
        return self.search.run(query)

    def direct_gemini_query(self, question: str):
        """Direct query to Gemini without web search"""
        response = self.llm.invoke([HumanMessage(content=question)])
        return response.content


def main():
    # Create the RAG system
    rag_system = SimpleRAGWithSerpAPI()

    # Example 1: Simple search
    print("=== Simple Search ===")
    result = rag_system.simple_search("What is Python programming?")
    print(f"Search Result: {result[:200]}...")

    # Example 2: Direct Gemini query
    print("\n=== Direct Gemini Query ===")
    direct_result = rag_system.direct_gemini_query("What is machine learning?")
    print(f"Gemini Response: {direct_result[:200]}...")

    print("\n=== RAG with Context ===")
    # Example 3: RAG with proper context
    question1 = "What are the latest AI developments in 2024?"
    print(f"\nQuestion: {question1}")
    response1 = rag_system.query_with_rag(question1)
    print(f"Answer: {response1['answer'][:300]}...")
    print(f"Context length: {response1['context_length']} characters")
    print("-" * 50)

    print("\n=== RAG with Memory ===")
    # Example 4: Follow-up question with memory
    question2 = "How do these developments compare to previous years?"
    print(f"\nQuestion: {question2}")
    response2 = rag_system.query_with_memory(question2)
    print(f"Answer: {response2['answer'][:300]}...")
    print(f"Has conversation history: {response2['has_history']}")
    print("-" * 50)

    # Example 5: Another follow-up
    question3 = "What are the potential applications of these AI developments?"
    print(f"\nQuestion: {question3}")
    response3 = rag_system.query_with_memory(question3)
    print(f"Answer: {response3['answer'][:300]}...")
    print(f"Has conversation history: {response3['has_history']}")


if __name__ == "__main__":
    main()
