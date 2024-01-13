import glob
import os
from llama_index import Prompt, VectorStoreIndex, ServiceContext, Document, StorageContext
from llama_index.llms import OpenAI
import pandas as pd
from llama_index import VectorStoreIndex, StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index.vector_stores import AstraDBVectorStore
from langchain.llms import OpenAI


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
    collection_name="tourism3",
    embedding_dimension=1024,
)

#AstraDB(str(astra_api_key), "https://9cebec1d-fe4a-40f7-8649-56b4b64fe1f5-us-east1.apps.astra.datastax.com").delete_collection(collection_name="tourism")


service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-1106", temperature=0.6, system_prompt=system_prompt),
                                               embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name='dangvantuan/sentence-camembert-large', model_kwargs = {'device': 'cuda:0'})))    

storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

if "y"==input("Do you want to recompute the index? (y/n)"):
    documents = []

    # Get a list of all CSV files in the directory
    csv_files = glob.glob('./data/*.csv')

    for file in csv_files:
        df = pd.read_csv(file, dtype=str, parse_dates=True)

        # Convert the DataFrame into a list of Document objects
        docs = [Document(doc_id=str(i), text=row.to_string(), metadata={'url': row['event']}) for i, row in df.iterrows()]

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
    # Use the memory object to query the index
    response = index.as_query_engine().query(question).response

    # Print the response
    print(response)


print("Goodbye!")