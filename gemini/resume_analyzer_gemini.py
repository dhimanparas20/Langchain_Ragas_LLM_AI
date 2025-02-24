#########################################################
# Analyzes Resume using Gemini API
#########################################################

import os
os.system("clear")
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.prompts import ChatPromptTemplate
from datasets import Dataset
import json
from langchain_community.vectorstores import Chroma
from time import sleep
import plotly.graph_objects as go
import numpy as np

#--------------------------------
#Setting Up Global Variables
#--------------------------------
BASE_FILE:str = "input/dataset/resume.pdf"
RAW_JSON_FILE:str = "input/json/rawSet_resume.json"
OUTPUT_DATA_JSON_FILE:str = "output/resume.json"
MODEL:str = "gemini-2.0-flash"  # Updated to latest model
TEMPERATURE:float = 0.2
ALREDY_HAVE_DATASET:bool = False
USE_CHROMA_DB:bool = True
TEXT_SPLITTER_CHUNK_SIZE:int = 1024
TEXT_SPLITTER_CHUNK_OVERLAP:int = 50

# Updated prompt template to use chat format
PROMPT_MESSAGES = [
    ("system", "You are an expert resume analyzer. Your task is to analyze resumes and provide detailed, accurate information based on the content provided."),
    ("human", "Context: {context}\nQuestion: {input}")
]

class ResumeAnalyzer:
    def __init__(self):
        self.total_questions = 0
        self.total_tokens = 0
        self.answers = []
        self.contexts = []
        
    def get_docs(self):
        print(f"Loading Base Input File: {BASE_FILE}")
        if ".pdf" in BASE_FILE:
            loader = PyPDFLoader(BASE_FILE)
        elif ".txt"in BASE_FILE:    
            loader = TextLoader(BASE_FILE)
        else:   
            print("Invalid File type to read from.")
            return None
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
            chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
        )
        return text_splitter.split_documents(docs)

    def create_vector_store(self, docs):
        print("Creating Vector Database")
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        if USE_CHROMA_DB:
            print("Using Chroma DB")
            vectorStore = Chroma.from_documents(
                documents=docs,
                embedding=embedding,
                collection_name="resume_analysis",
                persist_directory="./chroma_db"
            )
        else:  
            print("Using In Memory DB")  
            vectorStore = FAISS.from_documents(docs, embedding)
        return vectorStore

    def create_chain(self, retriever):
        print("Creating Retrieval Chain")
        
        # Initialize Gemini model with safety settings
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            temperature=TEMPERATURE,
            max_tokens=None,
            top_p=0.8,
            top_k=40,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        prompt = ChatPromptTemplate.from_messages(PROMPT_MESSAGES)
        
        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
        )
        
        return create_retrieval_chain(retriever, document_chain)

    def ask_question(self, chain, question, count):
        response = chain.invoke({
            "input": question
        })
        
        print("--------------------------------------")
        print(f"Question No.: {count}")
        
        # Track token usage if available
        if hasattr(response, 'usage_metadata'):
            self.total_tokens += response.usage_metadata.get('total_tokens', 0)
            
        self.total_questions += 1
        return response

    def analyze_resume(self):
        if not ALREDY_HAVE_DATASET:
            docs = self.get_docs()
            if not docs:
                return
                
            vectorStore = self.create_vector_store(docs)
            retriever = vectorStore.as_retriever()
            chain = self.create_chain(retriever)

            # Load questions
            with open(RAW_JSON_FILE, 'r') as file:
                test_set = json.load(file)

            # Process questions
            for count, question in enumerate(test_set['question'], 1):
                response = self.ask_question(chain, question, count)
                context = [doc.page_content for doc in response["context"]]
                self.answers.append(response['answer'])
                self.contexts.append(context)
                sleep(1)

            # Save results
            test_set['answer'] = self.answers
            test_set['contexts'] = self.contexts
            
            with open(OUTPUT_DATA_JSON_FILE, "w") as json_file:
                json.dump(test_set, json_file, indent=2)
                
            return test_set
        else:
            print(f"Reading existing dataset: {OUTPUT_DATA_JSON_FILE}")
            with open(OUTPUT_DATA_JSON_FILE, 'r') as file:
                return json.load(file)

    def generate_report(self, test_set):
        print("\n=========================================================")
        print("Analysis Report:")
        print(f"Total Questions Processed: {self.total_questions}")
        if self.total_tokens:
            print(f"Total Tokens Used: {self.total_tokens}")
        print("=========================================================\n")

        # Create visualization
        metrics = {
            "Questions Answered": self.total_questions,
            "Average Answer Length": np.mean([len(a.split()) for a in test_set['answer']]),
            "Context Usage": len(self.contexts),
            "Completion Rate": 1.0
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            name='Analysis Metrics'
        ))
        
        fig.update_layout(
            title='Resume Analysis Metrics',
            xaxis_title='Metrics',
            yaxis_title='Values',
            width=800,
        )

        fig.show()

def main():
    analyzer = ResumeAnalyzer()
    test_set = analyzer.analyze_resume()
    if test_set:
        analyzer.generate_report(test_set)

if __name__ == "__main__":
    main()
