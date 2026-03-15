import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
import chromadb
import chromadb.utils.embedding_functions

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

folder_path = "Thyroid"

documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents.")
# Step 2: Chunk it

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

"""
Embeddings convert text → numbers (vectors) so we can do similarity search. 
ChromaDB stores those vectors and lets us query "find me the most similar 
chunks to this question."
"""
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Init ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="thyroid_docs")

# Embed + store
contents = [c["content"] for c in chunks]
sources = [c["source"] for c in chunks]
embeddings = embedder.encode(contents).tolist()

collection.add(
    documents=contents,
    embeddings=embeddings,
    metadatas=[{"source": s} for s in sources],
    ids=[str(i) for i in range(len(contents))]
)

print(f"Stored {collection.count()} chunks in ChromaDB")
# Step 4: Retrieve relevant chunks

def retrieve_chunks(query, collection, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]  # top 3 relevant chunks

# Step 5: Send to Groq + get answer

def ask_groq(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question.
    
Context:
{context}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.9
)
    return response.choices[0].message.content

# Run it!
query = "What are the symptoms of thyroid disease?"
chunks = retrieve_chunks(query, collection)
answer = ask_groq(query, chunks)
print(answer)
