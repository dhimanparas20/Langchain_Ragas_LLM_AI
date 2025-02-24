from dotenv import load_dotenv
from os import system
system("clear")
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
# from pydantic.v1 import BaseModel
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
import time
import os

# Load environment variables
load_dotenv()

# Set USER_AGENT environment variable if not already set
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "MyLangChainApp/1.0"

# Initialize chat history
chat_history = []

# Function to load documents from a PDF file
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

# Function to load documents from a text file
def load_text(file_path):
    loader = TextLoader(file_path)
    docs = loader.load()
    return docs

# Function to load documents from a webpage
def load_webpage(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

# Function to split documents into smaller chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)
    return split_docs

# Function to create a vector store using Gemini embeddings
def create_vector_store(docs, use_chrome_db=False):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if use_chrome_db:
        # Use Chroma for storing embeddings
        print("Using Chroma for vector storage.")
        vector_store = Chroma(
            collection_name="foo",
            embedding_function=embedding,
            persist_directory="./chroma_langchain_db",  # Where to save data locally
        )
        vector_store.add_documents(documents=docs)
    else:
        # Use FAISS for storing embeddings
        print("Using FAISS for vector storage.")
        vector_store = FAISS.from_documents(docs, embedding=embedding)
    
    return vector_store

# Function to create a conversational chain
def create_chain(vector_store):
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.4,
        top_p=1,
        top_k=32,
        max_output_tokens=2048,
    )

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a document analyzer. Answer any question based on the provided context."),
        ("human", """
        Conversation history:
        {history}
        
        Context: {context}
        Question: {input}
        """)
    ])

    # Create a document chain
    document_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)

    # Create a retrieval chain
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# Function to handle user questions
def ask_question(chain, question):
    # Prepare the conversation history for the prompt
    history_text = "\n".join(chat_history)

    # Track time for approximate token estimation
    start_time = time.time()

    response = chain.invoke({
        "input": question,
        "history": history_text
    })

    end_time = time.time()

    print("--------------------------------------")
    # Estimate tokens based on response length
    approximate_tokens = len(str(response)) / 4  # rough estimation
    print(f"Approximate Tokens Used: {int(approximate_tokens)}")
    print(f"Response Time: {end_time - start_time:.2f} seconds")
    print("--------------------------------------")

    # Store the question and response in chat history
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {response}")

    return response

# Main function to handle user input and process documents
def main():
    print("Welcome! Please provide a file path (.pdf or .txt) or a webpage URL.")
    input_source = input("Enter file path or URL: ").strip()

    # Ask the user whether to use Chroma or FAISS
    use_chrome_db = input("Do you want to use Chroma for vector storage? (yes/no): ").strip().lower() == "yes"

    # Load documents based on input type
    try:
        if input_source.endswith(".pdf"):
            print("Loading PDF document...")
            docs = load_pdf(input_source)
        elif input_source.endswith(".txt"):
            print("Loading text document...")
            docs = load_text(input_source)
        elif input_source.startswith("http://") or input_source.startswith("https://"):
            print("Loading webpage content...")
            docs = load_webpage(input_source)
        else:
            print("Invalid input. Please provide a valid file path or URL.")
            return

        # Split documents into smaller chunks
        print("Splitting documents into chunks...")
        split_docs = split_documents(docs)

        # Create vector store and chain
        print("Creating vector store...")
        vector_store = create_vector_store(split_docs, use_chrome_db=use_chrome_db)
        print("Initializing chain...")
        chain = create_chain(vector_store)

        print("Ready for questions! (Type 'exit' or 'quit' to end)")

        # Interaction loop
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            answer = ask_question(chain, user_input)
            print(f"\nAssistant: {answer['answer']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please try again.")

# Run the main function
if __name__ == "__main__":
    main()
