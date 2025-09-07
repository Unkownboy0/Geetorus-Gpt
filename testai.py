import fitz  # PyMuPDF
import requests
from io import BytesIO
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def extract_text_from_pdf_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_document = fitz.open(stream=BytesIO(response.content), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
pdf_url = "https://docs.aws.amazon.com/pdfs/AmazonS3/latest/userguide/s3-userguide.pdf"
extracted_text = extract_text_from_pdf_url(pdf_url)

if not extracted_text:
    print("No text extracted from PDF.")
    exit()

text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([extracted_text])
print(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Embed the page_content of each Document
embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
print(embeddings)

# Initialize the vector store and add embeddings
vector_store = FAISS.from_documents(docs, embedding_model)

# Save the vector store locally
vector_store.save_local("example_index")

# Load the vector store
vector_store = FAISS.load_local("example_index", embedding_model, allow_dangerous_deserialization=True)

def answer_question(question, vector_store, embedding_model):
    # Perform similarity search using the vector store
    results = vector_store.similarity_search(question, k=3)
    return results

question = "What is the main topic of the document?"
answers = answer_question(question, vector_store, embedding_model)

print("Top 3 most relevant chunks for the question:")
for i, answer in enumerate(answers):
    print(f"Chunk {i+1}: {answer.page_content}")



