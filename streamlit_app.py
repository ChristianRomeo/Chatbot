# Import the necessary libraries
import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, StorageContext, set_global_service_context
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.node_parser import SentenceSplitter
from llama_index.indices.prompt_helper import PromptHelper
import re
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from langchain_openai import ChatOpenAI

# Streamlit interface
st.title('🦜🔗 Tourism Assistant Chatbot')

if "init" not in st.session_state:
    st.session_state.init = True
    system_prompt = (
    "You are a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
    )


    st.session_state.service_context = ServiceContext.from_defaults(llm=ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.9), 
                                                                    prompt_helper = PromptHelper(),
                                                                    embed_model= LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large')),
                                                                    node_parser=SentenceSplitter(),
                                                                    system_prompt=system_prompt,
                                                                    )

    set_global_service_context(st.session_state.service_context)

    # create or get a chroma collection
    st.session_state.chroma_collection = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection("tourism_db")

    # assign chroma as the vector_store to the context
    st.session_state.storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=st.session_state.chroma_collection))

    st.session_state.index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=st.session_state.chroma_collection), 
                                                                storage_context=st.session_state.storage_context, service_context=st.session_state.service_context)

    #context_prompt= "Base the reply to the user question mainly on the Description field of the context "
    #condense_prompt = " "

    #index.as_retriever(service_context=service_context, search_kwargs={"k": 1})

    st.session_state.retriever=VectorIndexRetriever(st.session_state.index, similarity_top_k=5) 
    
    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
                                                                                retriever=st.session_state.retriever, 
                                                                                query_engine=st.session_state.index.as_query_engine(service_context=st.session_state.service_context, 
                                                                                                                                    retriever=st.session_state.retriever ),
                                                                                service_context=st.session_state.service_context,
                                                                                system_prompt=system_prompt,
                                                                                #condense_prompt=DEFAULT_CONDENSE_PROMPT_TEMPLATE,
                                                                                #context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE,
                                                                                #verbose=True,
                                                                            )
        
    st.session_state.messages = []



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def handle_chat(question):
    if question.lower() == "reset":
        st.session_state.chat_engine.reset()
        st.session_state.messages = []
        return "The conversation has been reset."
    else:
        response = st.session_state.chat_engine.chat(question)
        cleaned_response = re.sub(r"(AI: |AI Assistant: |assistant: )", "", re.sub(r"^user: .*$", "", str(response), flags=re.MULTILINE))
        return cleaned_response

user_input = st.chat_input("Please enter your question:")
if user_input:
    if user_input.lower() == "exit":
        st.stop()
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Handle chat and get the response
        response = handle_chat(user_input)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Update the chat messages displayed
        st.rerun()