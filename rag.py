import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 1: Load doc

def load_documents(folder_path):
    docs = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    docs.append({"content": page.page_content, "source": file_path})
    return docs

folder_path = "C:\\Users\\Pooja\\Azure_Project\\Thyroid"

documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents.")
# Step 2: Chunk it

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append({"content": split, "source": doc["source"]})
    return chunks

chunks = chunk_documents(documents)
print(f"Total chunks: {len(chunks)}")
# Step 3: Embed + store in ChromaDB
# Step 4: Retrieve relevant chunks
# Step 5: Send to Groq + get answer