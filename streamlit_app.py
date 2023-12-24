# Import the necessary libraries
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.vector_stores import AstraDBVectorStore

# Set the title of the app
st.title("Tourism Assistant")

# Set the description of the app
system_prompt = (
    "This is a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
)

# Set up the database connection
astra_db_store = AstraDBVectorStore(
    token=str(st.secrets["ASTRA_API_KEY"]),
    api_endpoint=str(st.secrets["api_endpoint"]),
    collection_name="tourism2",
    embedding_dimension=1024,
)

# Set up the model and embedding
service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.6, system_prompt=system_prompt),
                                               embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large')))
 
storage_context = StorageContext.from_defaults(vector_store=astra_db_store)
index = VectorStoreIndex.from_vector_store(vector_store=astra_db_store, service_context=service_context) 

# Create a text input for the user to enter their question
question = st.text_input("Please enter your question: ")

# When the question is entered, query the model and display the result
if question:
    result = index.as_query_engine().query(question)
    st.write(result)