
import os
import gradio as gr

from functools import lru_cache
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load AI model only when needed
@lru_cache(maxsize=1)
def get_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )


# Load embedding model only when needed
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# Process uploaded HR policy PDF
@lru_cache(maxsize=2)
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    return vector_db


# Chatbot question-answering function
def hr_chatbot(message, history, pdf_file):

    if not pdf_file:
        return "Please upload an HR policy PDF first."

    if not message or not message.strip():
        return "Please enter an HR policy question."

    try:
        # Retrieve relevant document information
        vector_db = load_pdf(pdf_file)

        docs = vector_db.similarity_search(
            message,
            k=2
        )

        if not docs:
            return "No relevant information found in the HR policy."

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        # Generate prompt
        prompt = f"""Answer using only the HR policy context.
If the answer is not available, say:
I could not find that information in the HR policy.

Context:
{context[:1500]}

Question: {message}
Answer:"""

        # Generate AI response
        generator = get_generator()

        result = generator(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()

        if not answer:
            answer = "I could not find that information in the HR policy."

        # Show retrieved document pages
        pages = sorted(set(
            doc.metadata.get("page", 0) + 1
            for doc in docs
        ))

        page_text = ", ".join(map(str, pages))

        return (
            answer
            + "\n\nRetrieved source pages: "
            + page_text
        )

    except Exception as e:
        print("Chatbot error:", repr(e))
        return "Unable to process the document. Please check the server logs."


# PDF upload component
pdf_input = gr.File(
    label="Upload Nestle HR Policy PDF",
    file_types=[".pdf"],
    type="filepath"
)


# Create chatbot interface
demo = gr.ChatInterface(
    fn=hr_chatbot,
    additional_inputs=[pdf_input],
    title="Nestlé AI-Powered HR Assistant",
    description="Upload an HR policy PDF and ask questions.",
    examples=[
        ["What is Nestlé's HR policy about?", None],
        ["What are the HR principles?", None],
        ["What does the policy say about employees?", None]
    ]
)


# Render deployment configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False
    )
