import fitz  # PyMuPDF
import requests
from io import BytesIO
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini import

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

# Initialize Gemini model
gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-pro"
    google_api_key="YOUR_GEMINI_API_KEY",  # Replace with your Gemini API key
    temperature=0.3,
    max_tokens=1024,
)

def answer_question_with_gemini(question, vector_store, embedding_model, gemini):
    # Retrieve relevant chunks
    results = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Use Gemini to generate the answer
    response = gemini.invoke(prompt)
    return response

question = "What is the main topic of the document?"
answer = answer_question_with_gemini(question, vector_store, embedding_model, gemini)

print("Gemini's answer:")
print(answer)



