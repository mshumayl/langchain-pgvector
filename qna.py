import time
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import OpenAI

def process_data():
    print("Getting data...")
    loader = DirectoryLoader("documents", glob="**/*.txt")
    documents = loader.load()
    print("Documents loaded.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    print("Text splitted.")
    
    texts = text_splitter.split_documents(documents)
    
    print(texts)
    
    return texts

def get_embeddings(texts):
    print("Getting embeddings...")
    embeddings = OpenAIEmbeddings()
    print("Embeddings loaded.")
    
    print("Creating vector store...")
    vector_store = Chroma.from_documents(texts, embeddings)
    print("Vector store created.")
    
    qna = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qna

def query(prompt, qna_retriever):
    print(f"Answer: {qna_retriever.run(prompt)}")

if __name__ == '__main__':
    load_dotenv()
    texts = process_data()
    qna_retriever = get_embeddings(texts)
    
    while True:
        prompt = input("Prompt: ")
        
        if prompt == "":
            break
        
        query(prompt, qna_retriever)
        cont = input("Press 'Enter' to prompt again.")
        
        if cont != "":
            break
