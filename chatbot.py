import streamlit as st
from langchain_community.llms import HuggingFaceHub 
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import io
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter

load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"


def get_pdf_text(uploaded_file):
    text = ''
    with io.BytesIO(uploaded_file.read()) as file:
        reader = PdfReader(file)
        for page_num in range(len(reader.pages)) :
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def get_text_spliter(text):
    text_split = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    return text_split.split_text(text)

def get_embedding_vector(chunk):
    embed = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunk,embedding=embed)
    vectorstore.save_local("pdf_index")
    
def question_answer_chain():
    
    
    
    prompt_template = """
        Answer the question as precisely as possible based on the provided context.
        Ensure that your answer is accurate. If the answer is not available, simply state "I don't know the answer"
        instead of providing incorrect information.
    
        Context:
        {context}?

        Question:
        {question}

        Answer:
        """
         
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                           repo_id =model_id,
                           model_kwargs={'temperature':0.3}
                                      )
    
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context','question']
                            )
    
    chain = load_qa_chain(llm=model,prompt=prompt)
    
    return chain


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings()
    
    new_db = FAISS.load_local("pdf_index",embeddings)
    
    docs = new_db.similarity_search(user_question)
    
    chain = question_answer_chain()
    
    response = chain(
        {"input_documents":docs,
         "question":user_question},
        return_only_outputs =True
    )
    
    print(response)
    
    st.write("Reply:",response['output_text'])



def main():
    # Set page configuration
    st.set_page_config(page_title="Chatbot | Amphibius", layout="wide", initial_sidebar_state="collapsed")
    
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
        .header {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .sidebar {
            background-color: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
        }
        .content {
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<div class='header'>Chat with your PDF file and Translate Text </div>", unsafe_allow_html=True)
    
    # File uploader
    st.markdown("<div class='content'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload your PDF:")
    question = st.text_input("Ask a question related to your PDF?")
    
    if question:
        user_input(question)
        
    st.selectbox("Select a desired language to get answer:", ("English", "Hindi", "Marathi", "French", "German", "Tamil"))
    if st.button("Translate and Display the answer"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(file)
            chunks = get_text_spliter(raw_text)
            get_embedding_vector(chunks)
 
            st.success("Done")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
        st.title("PDF based LLM-LANGCHAIN Chatbot & Translator App")
        st.subheader("About App:")
        st.write("The app's primary resource is utilised to create:")
        st.markdown("- streamlit")
        st.markdown("- langchain")
        st.markdown("- Gemini / HuggingFace")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
