from dotenv import load_dotenv
from os import system, path
import time
import shutil
from typing import List, Dict, Any
import os

# Load environment variables and clear screen
load_dotenv()
system("clear")

# Import required libraries
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains import create_retrieval_chain

# Configuration
USE_CHROMA_DB = True  # Set to False to use FAISS instead
CHROMA_DB_DIR = "./chroma_db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_document(self, file_path: str) -> List[Any]:
        """Load and split document based on file type"""
        try:
            if file_path.startswith(('http://', 'https://')):
                print("Loading web content...")
                loader = WebBaseLoader(file_path)
            elif file_path.endswith('.pdf'):
                print("Loading PDF document...")
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                print("Loading text file...")
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file format. Please use PDF, TXT, or URL")

            documents = loader.load()
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Document split into {len(split_docs)} chunks")
            return split_docs

        except Exception as e:
            print(f"Error loading document: {str(e)}")
            return []

class VectorStoreManager:
    def __init__(self):
        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

    def create_vector_store(self, docs: List[Any], collection_name: str = "default") -> Any:
        """Create and return vector store"""
        try:
            if USE_CHROMA_DB:
                if path.exists(CHROMA_DB_DIR):
                    shutil.rmtree(CHROMA_DB_DIR)
                
                vector_store = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embedding,
                    persist_directory=CHROMA_DB_DIR,
                    collection_name=collection_name
                )
                vector_store.persist()
                print(f"Chroma DB created at: {CHROMA_DB_DIR}")
            else:
                vector_store = FAISS.from_documents(docs, self.embedding)
                print("FAISS vector store created in memory")
            
            return vector_store

        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return None

    def load_existing_store(self, collection_name: str = "default") -> Any:
        """Load existing vector store if available"""
        try:
            if USE_CHROMA_DB and path.exists(CHROMA_DB_DIR):
                vector_store = Chroma(
                    persist_directory=CHROMA_DB_DIR,
                    embedding_function=self.embedding,
                    collection_name=collection_name
                )
                return vector_store
            return None
        except Exception as e:
            print(f"Error loading existing vector store: {str(e)}")
            return None

class ChatBot:
    def __init__(self):
        self.chat_history = []
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            top_p=1,
            top_k=32,
            max_output_tokens=2048,
        )

    def create_chain(self, vector_store: Any):
        """Create conversation chain"""
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant specialized in analyzing documents 
            and providing accurate, relevant information based on the content provided. 
            Always cite specific parts of the document when possible."""),
            ("human", """
            Previous conversation:
            {history}
            
            Context from document:
            {context}
            
            Question: {input}
            """)
        ])

        document_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompt_template
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        return create_retrieval_chain(retriever, document_chain)

    def process_question(self, chain, question: str) -> Dict[str, Any]:
        """Process question and return response with metrics"""
        history_text = "\n".join(self.chat_history[-6:])  # Keep last 3 QA pairs
        
        start_time = time.time()
        response = chain.invoke({
            "input": question,
            "history": history_text
        })
        end_time = time.time()

        # Update chat history
        self.chat_history.append(f"Q: {question}")
        self.chat_history.append(f"A: {response['answer']}")

        return {
            "answer": response['answer'],
            "response_time": end_time - start_time,
            "tokens_estimate": len(str(response)) // 4
        }

def print_response_metrics(metrics: Dict[str, Any]):
    """Print formatted response metrics"""
    print("\n" + "─" * 50)
    print(f"⏱️  Response Time: {metrics['response_time']:.2f} seconds")
    print(f"📊 Estimated Tokens: {metrics['tokens_estimate']}")
    print("─" * 50 + "\n")

def main():
    print("🤖 Initializing Document Q&A System...")
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store_manager = VectorStoreManager()
    chatbot = ChatBot()

    # Check for existing vector store
    if USE_CHROMA_DB:
        existing_store = vector_store_manager.load_existing_store()
        if existing_store:
            use_existing = input("📚 Found existing document database. Use it? (y/n): ").lower()
            if use_existing == 'y':
                chain = chatbot.create_chain(existing_store)
                print("\n✨ Ready to answer questions about the existing document!")
                interactive_qa_session(chatbot, chain)
                return

    # Process new document
    while True:
        file_path = input("\n📄 Enter document path (PDF/TXT) or URL: ").strip()
        if path.exists(file_path) or file_path.startswith(('http://', 'https://')):
            break
        print("❌ Invalid path or URL. Please try again.")

    # Process document and create vector store
    docs = doc_processor.load_document(file_path)
    if not docs:
        print("❌ Document processing failed. Exiting...")
        return

    vector_store = vector_store_manager.create_vector_store(docs)
    if not vector_store:
        print("❌ Vector store creation failed. Exiting...")
        return

    chain = chatbot.create_chain(vector_store)
    print("\n✨ Document processed! Ready for questions!")
    interactive_qa_session(chatbot, chain)

def interactive_qa_session(chatbot: ChatBot, chain):
    """Handle interactive Q&A session"""
    print("\n💡 Type 'quit' or 'exit' to end the session")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("❌ Please enter a valid question.")
                continue
            
            response = chatbot.process_question(chain, question)
            print(f"\n🤖 Answer: {response['answer']}")
            print_response_metrics(response)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
