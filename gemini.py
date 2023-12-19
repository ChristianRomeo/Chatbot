import glob
import os
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
import pandas as pd
import google.generativeai as genai
import google.ai.generativelanguage as glm
import dask.dataframe as dd


class GeminiEmbeddingFunction(EmbeddingFunction):
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
print(answer.text)
