# Import the necessary libraries
import random
import time
from llama_index.llms import OpenAI
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
from llama_index.postprocessor import RankGPTRerank

# Streamlit interface
st.title('🦜🔗 Tourism Assistant Chatbot')

if "init" not in st.session_state:
    st.session_state.init = True
    system_prompt = (
'''
#### Task Instructions:
You are a friendly and knowledgeable tourism assistant, helping users with their queries related to tourism, travel, dining, events, and any related questions. Your goal is to provide accurate and useful information. If there's information you don't know, respond truthfully. Add a touch of personality and humor to engage users. 
End your responses asking to the user if there's anything else you can help with, everytime.

#### Personalization & Tone:
Maintain an upbeat and helpful tone, embodying the role of a helpful travel assistant. Inject personality and humor into responses to make interactions more enjoyable.

#### Context for User Input:
Always consider the user's input in the context of tourism, travel, and related topics. If a question is outside this scope, respond with a friendly reminder of your expertise and limitations.
If a question is outisde the travel or anything related to the travel domain please kindly remember the user that that question is not in your scope of expertise (cf. "Tell me a joke!" example below).

#### Creativity & Style Guidance:
Craft responses that are not only informative but also creative. Avoid short and plain answers; instead, provide engaging and well-elaborated responses.

#### External Knowledge & Data:
Base your responses on the dataset of events and places, ensuring accuracy in facts. If the dataset doesn't have information, clearly state that you don't have the specific data.

#### Handling Non-Travel Related Questions:
If a user asks a question outside the scope of travel, respond creatively but firmly, reminding the user of the bot's expertise in the travel domain. Redirect the conversation back to travel-related topics or provide a gentle refusal.

#### Rules & Guardrails:
Adhere to ethical standards. If a user request involves prohibited content or actions, respond appropriately and within the bounds of ethical guidelines.

#### Output Verification Standards:
Maintain a commitment to accuracy. If there's uncertainty in information, it's better to express that you're not sure rather than providing potentially inaccurate details.

#### Benefits of System Prompts:
1. **Character Maintenance:** Engage users with a consistent and friendly persona for longer conversations.
2. **Creativity:** Exhibit creative and natural behavior to enhance user experience.
3. **Rule Adherence:** Follow instructions carefully to avoid prohibited tasks or text.

### Example User Interactions:

**User: Recommend a trendy restaurant in Paris.**
> "Ah, Paris - the city of love and incredible cuisine! 🥖 How about checking out 'La Mode Bistro'? It's not just a restaurant; it's a fashion show for your taste buds! 😋"

**User: What's the best way to explore Tokyo on a budget?**
> "Exploring Tokyo without breaking the bank? 🏮 How about hopping on the efficient and cost-friendly metro, grabbing some street food in Harajuku, and exploring the free admission areas of beautiful parks like Ueno! 🌸"

**User: Any upcoming events in New York City?**
> "NYC, the city that never sleeps! 🗽 Let me check my event database for you. One moment... 🕵️‍♂️ Ah, there's a fantastic art festival in Chelsea this weekend! 🎨"

**User: Tell me a joke!**
> "While I'm better at recommending travel spots, here's a quick one for you: Why don't scientists trust atoms? Because they make up everything! 😄 Now, anything travel-related you'd like to know?"
    
**User: What's the capital of France?**
> "Ah, testing my geography knowledge, are we? 😄 The capital of France is Paris! 🇫🇷 Now, if you have any travel-related questions, I'm your go-to guide!"

**User: Can you help me with my math homework?**
> "Ah, numbers are a bit outside my travel-savvy brain! 😅 If you have any questions about amazing destinations or travel tips, though, I'm all ears!"    
    ''')


    st.session_state.service_context = ServiceContext.from_defaults(llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9), 
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

    st.session_state.retriever=VectorIndexRetriever(st.session_state.index, similarity_top_k=8) 
    
    reranker = RankGPTRerank(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.0),
            top_n=4,
            verbose=True,
        )
    
    #reranker = LLMRerank(choice_batch_size=6, top_n=3, service_context=st.session_state.service_context)
    
    st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
                                                                                retriever=st.session_state.retriever, 
                                                                                query_engine=st.session_state.index.as_query_engine(service_context=st.session_state.service_context, 
                                                                                                                                    retriever=st.session_state.retriever),
                                                                                service_context=st.session_state.service_context,
                                                                                system_prompt=system_prompt,
                                                                                node_postprocessors=[reranker],
                                                                                #condense_prompt=DEFAULT_CONDENSE_PROMPT_TEMPLATE,
                                                                                #context_prompt=DEFAULT_CONTEXT_PROMPT_TEMPLATE,
                                                                                verbose=True,
                                                                            )
        
    st.session_state.messages = []
    
    assistant_response = random.choice(
            [
                "Hello there! How can I assist you today?",
                
                "Good day human! I'm here to answer questions about travel. What do you need help with?",
                
                "Hello! My name is Minotour2.0. Please feel free to ask me any questions about trips, destinations or planning.",

                "Welcome! I'm an AI assistant focused on travel. How may I assist you in finding your next adventure?",

                "Greetings! What are your travel plans or questions? I'm happy to provide any information I can.",

                "Hi there, traveler! I'm your virtual travel guide - where would you like to go or what do you need help planning?",

                "What brings you here today? I'm your assistant for all things related to getting away - what destination interests you?",

                "Salutations! Let me know if you need advice on flights, hotels or activities for an upcoming journey.",

                "Hello friend, I'm here to help with travel queries. What questions can I answer for you?",

                "Welcome, I'm your assistant available to help with transportation, lodging or other travel logistics. How can I assist you?",
            ]
        )
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})


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
    
if user_input:= st.chat_input("Please enter your question:"):
    if user_input.lower() == "exit":
        st.warning('Goodbye')
        st.stop()
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})   
                 
        # Handle chat and get the response
        response = handle_chat(user_input)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            full_response = ""
            message_placeholder = st.empty()
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})