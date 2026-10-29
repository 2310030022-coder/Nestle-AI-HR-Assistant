
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Nestle AI HR Assistant",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Nestlé AI HR Assistant")
st.caption("Interactive HR policy document assistant")
st.write("Upload the Nestlé HR policy PDF and ask questions.")

@st.cache_data
def process_pdf(pdf_bytes):
    from io import BytesIO

    reader = PdfReader(BytesIO(pdf_bytes))
    passages = []
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text)

        sentences = re.split(r"(?<=[.!?])\s+", text)

        for i in range(0, len(sentences), 3):
            passage = " ".join(sentences[i:i+3]).strip()
            if len(passage) > 35:
                passages.append(passage)
                pages.append(page_num)

    return passages, pages

uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        passages, pages = process_pdf(uploaded_file.getvalue())

        if not passages:
            st.warning("No readable text found in this PDF.")
            st.stop()

        st.success(
            f"PDF processed successfully! "
            f"{len(passages)} searchable passages created."
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask about the HR policy...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2)
            )

            matrix = vectorizer.fit_transform(
                passages + [question]
            )

            scores = cosine_similarity(
                matrix[-1],
                matrix[:-1]
            )[0]

            top_indices = scores.argsort()[-3:][::-1]

            if scores[top_indices[0]] < 0.04:
                answer = (
                    "I could not find relevant information "
                    "in the uploaded HR policy."
                )
            else:
                answer = "### Relevant policy information\n\n"

                for index in top_indices:
                    if scores[index] >= 0.04:
                        answer += (
                            f"**Page {pages[index]}:** "
                            f"{passages[index]}\n\n"
                        )

                answer += (
                    "_These are retrieved document passages, "
                    "not AI-generated policy interpretations._"
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

    except Exception as e:
        st.error(f"Could not process PDF: {e}")

else:
    st.info("Please upload a PDF to begin.")
