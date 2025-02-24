from dotenv import load_dotenv
load_dotenv()
from os import system
system("clear")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader  # Using PyPDFLoader for PDF extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback

# Initialize chat history
chat_history = []

# Retrieve Data from PDF
def get_docs():
    loader = PyPDFLoader('input/dataset/resume.pdf')  # Load the PDF file
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = text_splitter.split_documents(docs)
     
    return splitDocs

def create_vector_store(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = ChatOpenAI(
        temperature=0.4,
        model='gpt-3.5-turbo-1106'
    )

    prompt_template = ChatPromptTemplate.from_template("""
    You are resume analyzer.Answer any question based on the resume provided. Here is the conversation history:
    {history}
    
    Context: {context}
    Question: {input}
    """)

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
    with get_openai_callback() as cb:
        response = chain.invoke({
            "input": question,
            "history": history_text  # Pass the chat history to the model
        })
    print("--------------------------------------")    
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")    
    print("--------------------------------------")    
    
    # Store the question and response in chat history
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {response}")

    return response

# Load documents and create vector store and chain
docs = get_docs()
vectorStore = create_vector_store(docs)
chain = create_chain(vectorStore)

# Example interaction loop
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit']:
        break
    
    answer = ask_question(chain, user_input)
    
    print(f"Assistant: {answer['answer']}")