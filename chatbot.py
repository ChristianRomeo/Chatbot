import glob
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext, set_global_service_context
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
import chromadb
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
import pandas as pd
import os
import numpy as np
from llama_index.node_parser import SentenceSplitter
from llama_index.indices.prompt_helper import PromptHelper
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from llama_index.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from summarizer.sbert import SBertSummarizer
from sentence_transformers import SentenceTransformer
from langchain.agents import initialize_agent
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms import OpenAI
from langchain.chains import ConversationChain
import chromadb.utils.embedding_functions as embedding_functions
import ray
import re
import torch
from llama_index.llms import HuggingFaceLLM
from transformers import pipeline
from langchain_community.llms import GPT4All, GooglePalm
from langchain_google_genai import GoogleGenerativeAI
from gpt4all import Embed4All
from llama_index.llms import ChatMessage, MessageRole
from llama_index.chat_engine import CondenseQuestionChatEngine, CondensePlusContextChatEngine
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever, VectorIndexAutoRetriever
from langchain_community.llms import Cohere
from langchain_community.chat_models import ChatCohere
import cohere

system_prompt = (
    "You are a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
)


llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.6)
#GPT4All(model="./mistral-7b-openorca.Q4_0.gguf", device='nvidia', n_threads=12, use_mlock=True, n_predict= 2000, temp=1) 
#ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.6)
#GoogleGenerativeAI(model="gemini-pro")

service_context = ServiceContext.from_defaults(llm= llm,
                                                #prompt_helper=PromptHelper(),
                                                embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large',
                                                                                                      model_kwargs={'device': 'cuda:0'})),
                                                node_parser=SentenceSplitter(),
                                                system_prompt=system_prompt,
                                                #chunk_size_limit=4096,
                                                #query_wrapper_prompt=query_wrapper_prompt
                                                )

set_global_service_context(service_context)

db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("tourism_db")

# assign chroma as the vector_store to the context
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=chroma_collection))

'''documents = []

ray.init()
# Get a list of all CSV files in the directory
for file in glob.glob('./data/*.csv'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file, dtype=str, parse_dates=True)

    # Convert the DataFrame into a list of Document objects
    docs = [Document(doc_id=str(i), text=row.to_string(), extra_info={"url": row['url']}) for i, row in df.iterrows()] #str(row.to_dict())

    # Add the documents to the list
    documents.extend(docs)

batch_size = 5461  # Maximum batch size
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    # Now add the batch to the index
    index = VectorStoreIndex.from_documents(batch, service_context=service_context, storage_context=storage_context, show_progress=True)
    #index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=chroma_collection), storage_context=storage_context).refresh(batch)
    #storage_context.persist(persist_dir=f"./chroma_db")
storage_context.persist(persist_dir=f"./chroma_db")'''

index = VectorStoreIndex.from_vector_store(ChromaVectorStore(chroma_collection=chroma_collection), storage_context=storage_context, service_context=service_context)

context_prompt= "Base the reply to the user question mainly on the Description field of the context "

chatEngine = CondensePlusContextChatEngine.from_defaults(
    retriever=VectorIndexRetriever(index, similarity_top_k=5), #index.as_retriever(service_context=service_context, search_kwargs={"k": 1}),
    query_engine=index.as_query_engine(service_context=service_context, retriever=VectorIndexRetriever(index, similarity_top_k=5)),
    service_context=service_context,
    system_prompt=system_prompt,
    #verbose=True,
)

while True:
    question = input("Please enter your question: ")
    if question.lower() == "exit":
        break
    elif question.lower() == "reset":
        chatEngine.reset()
        print("The conversation has been reset.")
        continue
    else:
        response = chatEngine.chat(question)
        #index.as_chat_engine(chat_mode="condense_plus_context", similarity_top_k=1, service_context=service_context,memory=ConversationBufferMemory()).chat(question)
        #print(response)
        
        filtered_text = re.sub(r"^user: .*$", "", str(response), flags=re.MULTILINE)
        print(re.sub(r"(AI: |AI Assistant: |assistant: )", "", filtered_text))
        

print("Goodbye!")

'''# Conversation loop with conversation history
#conversation_memory = []

memory = ConversationSummaryBufferMemory(llm=llm)

# Conversation loop
while True:
    question = input("Please enter your question: ")
    if question.lower() == "exit":
        break
    elif question.lower() == "reset":
        #conversation_memory = []  # Clear the conversation history
        memory.clear()
        print("The conversation has been reset.")
        continue
    else:
        #summary = SBertSummarizer(model="dangvantuan/sentence-camembert-large").model(conversation_memory)

        #print(summary)
        
        #full_text = "Given this chat history: " + str(summary) + "\nReply to the following question: " + question
        
        
        full_text = "Given this chat history: " + memory.load_memory_variables({})['history'] + "\nReply to the following question: " + question
        
        #print("full text: ", full_text)
        
        #reply =  index.as_retriever(service_context=service_context).retrieve(full_text)
        
        #reply = index.as_query_engine(service_context=service_context, verbose=True, memory=memory).query(full_text)
        
        reply = index.as_chat_engine(chat_mode= "condense_question").chat_repl()
        
        print(reply)
        
        memory.save_context({"input": question}, {"output": str(reply)})
        
        #print(memory)
        
        print("memory: ", memory.load_memory_variables({})['history'])
        
        # Update conversation memory with the new output
        #conversation_memory.append(full_text + "New reply: " + str(reply))

    #print(index.as_query_engine(verbose=True, similarity_top_k=6, memory=memory).query(question))'''