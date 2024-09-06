from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(
     model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

try:
    loader = TextLoader("data.txt")
except Exception as e:
    print("Error while loading file=", e)

# Create Embeddings
embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

#Use a smaller chunk size to manage token limits
text_splitter = CharacterTextSplitter (chunk_size=703, chunk_overlap=100)

# Create the index with the specified embedding model and text splitter
index_creator = VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter
)
index = index_creator.from_loaders([loader])

# Query the index with the LLM
response = index.query("How can I provide feedback about the class and what is the name of this class?", llm=llm)

print (response)

