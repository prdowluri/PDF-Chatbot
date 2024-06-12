import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os
import base64

# Load environment variables
load_dotenv() 

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    context_window=3900,
    token=os.getenv("HF_TOKEN"),
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.1},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def data_ingestion():
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    chat_text_qa_msgs = [
    (
    "user",
    """You are a Q&A assistant. Created by Prudhvi Raj Dowluri [linkdin](https://www.linkedin.com/in/prudhvi-raj-dowluri-412616221/) an AI Engineer. Your primary objective is to provide accurate and helpful answers based on the instructions and context provided.If a question falls outside the given context or scope, kindly guide the user to ask questions that align with the provided context.
    Context:
    {context_str}
    Question:
    {query_str}
    """
)
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."

# Streamlit app initialization
st.title("Your PDF Assistant 📄")
st.markdown("Get insights from your data – just chat!👇")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'I can answer your questions about a PDF. Just upload it!'}]

with st.sidebar:
    st.markdown("**Created by [Prudhvi](https://www.linkedin.com/in/prudhvi-raj-dowluri-412616221/)**")
    st.title(':blue[Get Started]:')
    uploaded_file = st.file_uploader("Upload your PDF and Click Submit")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            filepath = "data/saved_pdf.pdf"
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # displayPDF(filepath)  # Display the uploaded PDF
            data_ingestion()  # Process PDF every time new file is uploaded
            st.success("Done")

user_prompt = st.chat_input("Ask me anything about the data inside the document:")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = handle_query(user_prompt)
    st.session_state.messages.append({'role': 'assistant', "content": response})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

    