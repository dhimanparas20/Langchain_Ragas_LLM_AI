from dotenv import load_dotenv
load_dotenv()
from os import system
system("clear")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
import time

# Initialize chat history
chat_history = []

# Retrieve Data from PDF
def get_docs():
    loader = PyPDFLoader('input/dataset/resume.pdf')
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = text_splitter.split_documents(docs)
     
    return splitDocs

def create_vector_store(docs):
    # Using Gemini embeddings instead of OpenAI
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.4,
        top_p=1,
        top_k=32,
        max_output_tokens=2048,
    )

    # Using the same prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are resume analyzer. Answer any question based on the resume provided."),
        ("human", """
        Conversation history:
        {history}
        
        Context: {context}
        Question: {input}
        """)
    ])

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt_template
    )

    retriever = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

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
    # Since Gemini doesn't provide token counts directly, we'll estimate based on characters
    approximate_tokens = len(str(response)) / 4  # rough estimation
    print(f"Approximate Tokens Used: {int(approximate_tokens)}")
    print(f"Response Time: {end_time - start_time:.2f} seconds")
    print("--------------------------------------")    
    
    # Store the question and response in chat history
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {response}")

    return response

def main():
    # Load documents and create vector store and chain
    print("Loading documents...")
    docs = get_docs()
    print("Creating vector store...")
    vectorStore = create_vector_store(docs)
    print("Initializing chain...")
    chain = create_chain(vectorStore)
    print("Ready for questions! (Type 'exit' or 'quit' to end)")

    # Example interaction loop
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            answer = ask_question(chain, user_input)
            print(f"\nAssistant: {answer['answer']}")
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
