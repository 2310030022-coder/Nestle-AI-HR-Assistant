
import gradio as gr
from functools import lru_cache
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load free AI model
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

@lru_cache(maxsize=4)
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

def hr_chatbot(message, history, pdf_file):
    if not pdf_file:
        return "Please upload an HR policy PDF first."

    if not message.strip():
        return "Please enter a question."

    try:
        vector_db = load_pdf(pdf_file)

        docs = vector_db.similarity_search(message, k=2)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = f"""Answer using only the context.
If the answer is missing, say:
I could not find that information in the HR policy.

Context:
{context[:1500]}

Question: {message}
Answer:"""

        result = generator(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            truncation=True
        )

        answer = result[0]["generated_text"].strip()

        if not answer:
            answer = "I could not find that information in the HR policy."

        pages = sorted(set(
            doc.metadata.get("page", 0) + 1
            for doc in docs
        ))

        return (
            answer
            + "\n\nRetrieved source pages: "
            + ", ".join(map(str, pages))
        )

    except Exception as e:
        return f"Unable to process the document: {e}"

pdf_input = gr.File(
    label="Upload Nestle HR Policy PDF",
    file_types=[".pdf"],
    type="filepath"
)

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


import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
