import streamlit as st
from PyPDF2 import PdfReader

from langchain.embeddings import BedrockEmbeddings

from langchain.llms import Bedrock

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import BedrockChat
from langchain.schema import ChatMessage
import streamlit as st

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
     )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    if text_chunks is None:
        return FAISS(
            index=None, 
            docstore=None,
            index_to_docstore_id=None,
            embedding_function=embeddings.embed_documents,
        )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)



def main():
    st.set_page_config(page_title="Streamlit Question Answering App",
                       layout="wide",
                       page_icon=":books::parrot:")

    st.sidebar.markdown(
    """
    ### Instructions:
    1. Browse and upload PDF files
    2. Click Process
    3. Type your question in the search bar to get more insights
    """
)

    if "vectorDB" not in st.session_state:
        st.session_state.vectorDB = get_vectorstore(None)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state.messages:
        if msg.type == "human":
            st.chat_message("Human: ").write(msg.content)
        if msg.type == "ai":
            st.chat_message("Assistant: ").write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("Assistant"):
                stream_handler = StreamHandler(st.empty())

                llm = Bedrock(model_id="anthropic.claude-v2", streaming=True, callbacks=[stream_handler])

                conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorDB.as_retriever(search_kwargs={"k": 1}),
                )
                
                response = conversation_chain({'question': prompt, 'chat_history':st.session_state.messages})

                st.session_state.messages = st.session_state.messages + [HumanMessage(content = response["question"]), AIMessage(content = response["answer"])]
            
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                st.session_state.vectorDB = get_vectorstore(text_chunks)

                st.success('PDF uploaded successfully!', icon="✅")


if __name__ == '__main__':
    region_name ="us-east-1"
    model_id = "amazon.titan-embed-text-v1"

    embeddings = BedrockEmbeddings(
        region_name=region_name,
        model_id=model_id
    )                                    

main()