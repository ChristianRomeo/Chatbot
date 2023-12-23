import glob
import os
from llama_index import Prompt, VectorStoreIndex, ServiceContext, Document, StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
import pandas as pd
from llama_index import VectorStoreIndex, StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.vector_stores import AstraDBVectorStore

#OptimumEmbedding.create_and_save_optimum_model("dangvantuan/sentence-camembert-large", "./bge_onnx")

system_prompt = (
    "This is a knowledgeable Tourism Assistant designed to provide visitors with "
    "information, recommendations, and tips for exploring and enjoying their destination. "
    "The assistant is familiar with a wide range of topics including historical sites, "
    "cultural events, local cuisine, accommodations, transportation options, and hidden gems. "
    "It offers up-to-date and personalized information to help tourists make the most of their trip."
)

qa_template = Prompt((
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question and each answer should start with code word TourismBot: {query_str}\n"
))


astra_api_key = os.environ.get("ASTRA_API_KEY")

astra_db_store = AstraDBVectorStore(
    token=str(astra_api_key),
    api_endpoint="https://9cebec1d-fe4a-40f7-8649-56b4b64fe1f5-us-east1.apps.astra.datastax.com",
    collection_name="tourism2",
    embedding_dimension=1024,
)

service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.6, system_prompt=system_prompt), 
                                               embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large', model_kwargs = {'device': 'cuda:0'})))    
                                                #OptimumEmbedding(folder_name="./bge_onnx")
                                                #LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large', model_kwargs = {'device': 'cuda:0'}))
                                                #LangchainEmbedding(FastEmbedEmbeddings(model_name="intfloat/multilingual-e5-large", threads=15))
                                                #LangchainEmbedding(CohereEmbeddings(client=co, async_client=co, model='embed-multilingual-light-v3.0'))
                                                #LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large'))
                                                #LangchainEmbedding(VoyageEmbeddings(model='voyage-lite-01', show_progress_bar=True))
                                                #FastEmbedEmbedding(model_name="intfloat/multilingual-e5-large")
                                                #LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))) #.to('cuda'))

storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

if "y"==input("Do you want to recompute the index? (y/n)"):
    documents = []

    # Get a list of all CSV files in the directory
    csv_files = glob.glob('./data/*.csv')

    for file in csv_files:
        df = pd.read_csv(file, dtype=str, parse_dates=True)

        # Convert the DataFrame into a list of Document objects
        docs = [Document(doc_id=str(i), text=row.to_string()) for i, row in df.iterrows()]

        # Add the documents to the list
        documents.extend(docs)

    index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context, show_progress=True)
else:
    index = VectorStoreIndex.from_vector_store(vector_store=astra_db_store, service_context=service_context) 
    #load_index_from_storage(StorageContext.from_defaults(persist_dir=f"./storage"), service_context=service_context)

while True:
    # Read a line of input from the user
    question = input("Please enter your question: ")
    if question == "exit":
            break
    print(index.as_query_engine(verbose=True).query(question))

print("Goodbye!")