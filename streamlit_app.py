# Import the necessary libraries
import streamlit as st

from llama_index import VectorStoreIndex, ServiceContext, StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SentenceSplitter
from llama_index.indices.prompt_helper import PromptHelper
from langchain_community.llms import GPT4All
from llama_index import set_global_service_context
import re
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever



# Streamlit interface
st.title('Tourism Assistant Chatbot')


    
    
system_prompt = (
    "You are a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
)

llm = GPT4All(model="./mistral-7b-openorca.Q4_0.gguf", use_mlock=True, n_predict= 2000, temp=1)
#ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.9)
#GoogleGenerativeAI(model="gemini-pro")

service_context = ServiceContext.from_defaults(llm=llm, 
                                                prompt_helper = PromptHelper(),
                                                embed_model= LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large',model_kwargs = {'device': 'cuda:0'})),
                                                node_parser=SentenceSplitter(),
                                                system_prompt=system_prompt,
                                                #chunk_size_limit=4096
                                                )

set_global_service_context(service_context)

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("tourism_db")
#embedding_functions.HuggingFaceEmbeddingFunction(model_name="dangvantuan/sentence-camembert-large")

# assign chroma as the vector_store to the context
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=chroma_collection))

index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=chroma_collection), storage_context=storage_context, service_context=service_context)

#context_prompt= "Base the reply to the user question mainly on the Description field of the context "
#condense_prompt = " "

chatEngine = CondensePlusContextChatEngine.from_defaults(
    retriever=VectorIndexRetriever(index, similarity_top_k=1), #index.as_retriever(service_context=service_context, search_kwargs={"k": 1}),
    query_engine=index.as_query_engine(service_context=service_context, retriever=VectorIndexRetriever(index, similarity_top_k=1)),
    service_context=service_context,
    system_prompt=system_prompt,
    #condense_prompt=condense_prompt,
    #context_prompt=context_prompt,
    #verbose=True,
)

def handle_chat(question):
    if question.lower() == "reset":
        chatEngine.reset()
        return "The conversation has been reset."
    else:
        response = chatEngine.chat(question)
        cleaned_response = re.sub(r"(AI: |AI Assistant: |assistant: )", "", re.sub(r"^user: .*$", "", str(response), flags=re.MULTILINE))
        return cleaned_response
    
user_input = st.text_input("Please enter your question:")
if user_input:
    if user_input.lower() == "exit":
        st.stop()
    response = handle_chat(user_input)
    st.text(response)