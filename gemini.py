import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from llama_index.vector_stores import ChromaVectorStore
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from llama_index import VectorStoreIndex
from langchain.document_loaders import DirectoryLoader
import glob
from langchain_google_genai import GoogleGenerativeAI
from llama_index import VectorStoreIndex, ServiceContext, Document, StorageContext
import pandas as pd
import modin.pandas as mpd
import os
import chromadb.utils.embedding_functions as embedding_functions
import ray


# Set up the context
service_context = ServiceContext.from_defaults(llm= GoogleGenerativeAI(model="gemini-pro"), embed_model=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("tourism_events_gemini", embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ["GOOGLE_API_KEY"]))

# assign chroma as the vector_store to the context
storage_context = StorageContext.from_defaults(vector_store=ChromaVectorStore(chroma_collection=chroma_collection))

documents = []
ray.init()
# Get a list of all CSV files in the directory
for file in glob.glob('./data/*.csv'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file, dtype=str, parse_dates=True)

    # Convert the DataFrame into a list of Document objects
    docs = [Document(doc_id=str(i), text=str(row.to_dict()), metadata={"url": row['url']}) for i, row in df.iterrows()]

    # Add the documents to the list
    documents.extend(docs)

index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context, show_progress=True).refresh(documents)

conversation_memory = ConversationBufferMemory(return_messages=True)

# Conversation loop
while True:
    user_input = input("Please enter your question: ")
    if user_input.lower() == "exit":
        break
    elif user_input.lower() == "reset":
        conversation_memory.clear()  # Clear the conversation history
        print("The conversation has been reset.")
        continue
    else:
        # Generate the response using the index
        query = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(model="gemini-pro"), chain_type="stuff", retriever=index.as_retriever(), memory=conversation_memory)
        response = query({"question": user_input})

        # Save the response to the memory
        conversation_memory.save_context({"question": user_input}, {"answer": response["answer"]})

        # Print the response
        print(response["answer"])

'''class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/embedding-001'
    title = "Custom query"
    return genai.embed_content(model=model,
                               content=input,
                               task_type="retrieval_document",
                               title=title)["embedding"]

def create_chroma_db(documents, name):
  chroma_client = chromadb.Client()
  db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db

# Set up the DB
db = create_chroma_db(dd.read_csv(glob.glob('./data/*.csv'), dtype=str, encoding="mac_roman"), "googlecarsdatabase")

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage

# Perform embedding search
passage = get_relevant_passage("Exposition d'aquarelles", db)

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and conversational tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

prompt = make_prompt(input("query: "), passage)

model = genai.GenerativeModel('gemini-pro')
answer = model.generate_content(prompt)
print(answer.text)'''
