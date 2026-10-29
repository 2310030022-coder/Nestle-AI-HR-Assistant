
import streamlit as st
import re
import hashlib
from io import BytesIO
from collections import Counter

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Nestlé HR Intelligence Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PREMIUM INTERFACE
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #f2efff 0%, #e9f5ff 55%, #fff1f7 100%
    );
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eee8ff, #dcefff);
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(120deg, #4338ca, #7c3aed, #db2777);
    box-shadow: 0 10px 30px rgba(79,70,229,.18);
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 2.4rem;
}

.hero p {
    color: #f5f3ff !important;
    font-size: 1.05rem;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,.9);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #ddd6fe;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.9);
    border: 1px solid #ddd6fe;
    border-radius: 16px;
}

.stButton > button {
    border-radius: 12px;
    font-weight: 600;
}

.stDownloadButton > button {
    border-radius: 12px;
    background: #4f46e5;
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💼 Nestlé HR Intelligence Assistant</h1>
    <p>Explore, understand and search HR policies intelligently.</p>
    <p>
        📄 PDF Extraction &nbsp; | &nbsp;
        🧠 Document Summary &nbsp; | &nbsp;
        💬 Smart Q&A &nbsp; | &nbsp;
        🔎 Policy Search
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) >= 35
    ]


# ==========================================
# PDF DOCUMENT PROCESSING
# ==========================================

@st.cache_data(show_spinner=False)
def read_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    passages = []

    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text(extraction_mode="plain") or ""
        text = clean_text(raw)

        pages.append({
            "number": number,
            "text": text,
            "words": len(text.split())
        })

        sentences = make_sentences(text)

        for i in range(0, len(sentences), 3):
            passage = " ".join(sentences[i:i + 3]).strip()

            if len(passage) >= 40:
                passages.append({
                    "text": passage,
                    "page": number
                })

    return pages, passages


# ==========================================
# SEARCH ENGINE
# ==========================================

@st.cache_resource(show_spinner=False)
def create_index(document_hash, _texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(_texts)
    return vectorizer, matrix


def search_document(question, passages, vectorizer, matrix, limit=3):
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix)[0]

    ranked = scores.argsort()[::-1]
    matches = []

    for index in ranked:
        score = float(scores[index])

        if score < 0.035:
            continue

        matches.append({
            "page": passages[index]["page"],
            "text": passages[index]["text"],
            "score": score
        })

        if len(matches) >= limit:
            break

    return matches


# ==========================================
# EXTRACTIVE DOCUMENT SUMMARY
# ==========================================

def create_summary(pages, sentence_limit=6):
    candidates = []

    for page in pages:
        for sentence in make_sentences(page["text"]):
            if 45 <= len(sentence) <= 450:
                candidates.append({
                    "text": sentence,
                    "page": page["number"]
                })

    if not candidates:
        return []

    texts = [item["text"] for item in candidates]

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(texts)
        importance = matrix.mean(axis=0)

        # Score each sentence against the document's
        # overall vocabulary importance.
        scores = cosine_similarity(
            matrix,
            importance
        ).ravel()

        selected = scores.argsort()[::-1][:sentence_limit]
        selected = sorted(selected)

        return [candidates[i] for i in selected]

    except ValueError:
        return candidates[:sentence_limit]


# ==========================================
# SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("📁 Document Workspace")

    uploaded_pdf = st.file_uploader(
        "Upload your HR Policy PDF",
        type=["pdf"],
        help="Use a text-based PDF for best results."
    )

    st.divider()

    st.subheader("⚙️ Assistant Settings")

    result_count = st.slider(
        "Search results per question",
        1, 5, 3
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.suggested_question = None
        st.rerun()

    st.subheader("✨ Available Tools")
    st.write("💬 HR Policy Chat")
    st.write("📖 PDF Reader")
    st.write("🧠 Automatic Summary")
    st.write("🔎 Advanced Search")
    st.write("📥 Download Center")

    st.divider()
    st.caption("Python • Streamlit • PyPDF • Scikit-learn")
    st.caption("No paid API required")


# ==========================================
# WELCOME SCREEN
# ==========================================

if uploaded_pdf is None:
    st.info("👈 Upload a PDF in the sidebar to activate the assistant.")

    st.subheader("🚀 Your HR Document Intelligence Workspace")

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 📄 Extract")
        st.write("Turn PDF documents into clean, readable text.")

    with b:
        st.markdown("### 🧠 Summarize")
        st.write("Identify important policy sentences automatically.")

    with c:
        st.markdown("### 💬 Ask")
        st.write("Search HR policies using natural language.")

    st.stop()


# ==========================================
# LOAD DOCUMENT
# ==========================================

try:
    pdf_bytes = uploaded_pdf.getvalue()
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.document_hash != current_hash:
        st.session_state.document_hash = current_hash
        st.session_state.chat_history = []
        st.session_state.suggested_question = None

    with st.spinner("Processing your HR document..."):
        pages, passages = read_pdf(pdf_bytes)

    if not passages:
        st.warning(
            "No searchable text found. "
            "This PDF may be scanned or image-based."
        )
        st.stop()

    texts = tuple(item["text"] for item in passages)
    vectorizer, matrix = create_index(current_hash, texts)

except Exception as error:
    st.error("Could not process the uploaded PDF.")
    st.caption(str(error))
    st.stop()


# ==========================================
# DOCUMENT DASHBOARD
# ==========================================

st.success(f"✅ Loaded: {uploaded_pdf.name}")

total_words = sum(page["words"] for page in pages)

c1, c2, c3, c4 = st.columns(4)

c1.metric("📄 Pages", len(pages))
c2.metric("📝 Words", f"{total_words:,}")
c3.metric("🧩 Passages", len(passages))
c4.metric(
    "💬 Questions",
    sum(
        entry["role"] == "user"
        for entry in st.session_state.chat_history
    )
)

st.divider()

# ==========================================
# MAIN TABS
# ==========================================

tab_chat, tab_reader, tab_summary, tab_search, tab_download = st.tabs([
    "💬 Smart Chat",
    "📖 PDF Reader",
    "🧠 Summary",
    "🔎 Search",
    "📥 Downloads"
])


# ==========================================
# TAB 1: SMART CHAT
# ==========================================

with tab_chat:
    st.subheader("💬 Ask Your HR Policy Assistant")

    st.caption(
        "Ask a question and receive a concise document excerpt "
        "with supporting page references."
    )

    suggestions = [
        "What is Nestlé's HR policy about?",
        "What are the main HR principles?",
        "What does the policy say about employees?",
        "How does Nestlé support employee development?"
    ]

    left, right = st.columns(2)

    for i, suggestion in enumerate(suggestions):
        with (left if i % 2 == 0 else right):
            if st.button(
                "💡 " + suggestion,
                key=f"suggest_{i}",
                use_container_width=True
            ):
                st.session_state.suggested_question = suggestion

    st.divider()

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

            for source in entry.get("sources", []):
                with st.expander(f"📄 Source: Page {source['page']}"):
                    st.write(source["text"])

    typed = st.chat_input("Ask a question about the HR policy...")

    question = typed or st.session_state.suggested_question

    if question:
        st.session_state.suggested_question = None

        with st.chat_message("user"):
            st.write(question)

        matches = search_document(
            question,
            passages,
            vectorizer,
            matrix,
            result_count
        )

        if matches:
            best = matches[0]

            answer = (
                f"**Most relevant policy passage — Page {best['page']}**\n\n"
                f"{best['text']}\n\n"
                f"**Other relevant pages:** "
                + ", ".join(
                    str(p) for p in sorted({
                        item["page"] for item in matches
                    })
                )
                + "\n\n_This is extracted policy text, not a generated interpretation._"
            )
        else:
            answer = (
                "I couldn't find a relevant passage in the PDF. "
                "Try asking with different keywords."
            )

        with st.chat_message("assistant"):
            st.markdown(answer)

            for source in matches:
                with st.expander(f"📄 Page {source['page']} — View source"):
                    st.write(source["text"])

        st.session_state.chat_history.extend([
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": answer,
                "sources": matches
            }
        ])


# ==========================================
# TAB 2: PDF READER
# ==========================================

with tab_reader:
    st.subheader("📖 Clean PDF Text Reader")

    st.write(
        "Read extracted PDF text page by page. "
        "Useful for reviewing policies without opening the PDF separately."
    )

    page_choice = st.selectbox(
        "Choose a page",
        options=list(range(1, len(pages) + 1)),
        format_func=lambda x: f"Page {x}"
    )

    selected = pages[page_choice - 1]

    st.caption(f"{selected['words']} words on this page")

    st.text_area(
        "Extracted text",
        value=selected["text"],
        height=430
    )

    st.download_button(
        "📥 Download this page",
        data=selected["text"],
        file_name=f"nestle_hr_page_{page_choice}.txt",
        mime="text/plain",
        key="page_download"
    )

    with st.expander("📚 View all pages"):
        for page in pages:
            st.markdown(f"#### Page {page['number']}")
            st.text(page["text"][:12000])
            st.divider()


# ==========================================
# TAB 3: DOCUMENT SUMMARY
# ==========================================

with tab_summary:
    st.subheader("🧠 Automatic HR Policy Summary")

    st.info(
        "This is an extractive summary: important original "
        "sentences are selected from the document. "
        "It does not rewrite or independently interpret the policy."
    )

    summary_style = st.radio(
        "Choose summary length",
        ["Short", "Medium", "Detailed"],
        horizontal=True
    )

    summary_lengths = {
        "Short": 4,
        "Medium": 8,
        "Detailed": 14
    }

    selected_count = summary_lengths[summary_style]

    summary_items = create_summary(
        pages,
        sentence_limit=selected_count
    )

    if not summary_items:
        st.warning("No suitable sentences found for summarization.")
    else:
        summary_lines = []

        for item in summary_items:
            line = f"Page {item['page']}: {item['text']}"
            summary_lines.append(line)
            st.markdown(f"- **Page {item['page']}:** {item['text']}")

        summary_text = "\n\n".join(summary_lines)

        st.download_button(
            "📥 Download Summary",
            data=summary_text,
            file_name="Nestle_HR_Policy_Summary.txt",
            mime="text/plain"
        )


# ==========================================
# TAB 4: ADVANCED SEARCH
# ==========================================

with tab_search:
    st.subheader("🔎 Search the HR Policy")

    search_query = st.text_input(
        "Enter a keyword, topic, or question",
        placeholder="Example: employee development"
    )

    if search_query.strip():
        results = search_document(
            search_query,
            passages,
            vectorizer,
            matrix,
            10
        )

        if results:
            st.success(f"{len(results)} relevant passages found")

            for i, item in enumerate(results, 1):
                with st.expander(
                    f"Result {i} — Page {item['page']} "
                    f"— Relevance {item['score']:.0%}",
                    expanded=(i == 1)
                ):
                    st.write(item["text"])
                    st.caption(
                        "Relevance indicates text similarity, "
                        "not factual accuracy."
                    )
        else:
            st.warning("No relevant matches found.")


# ==========================================
# TAB 5: DOWNLOAD CENTER
# ==========================================

with tab_download:
    st.subheader("📥 Export Your HR Policy Analysis")

    full_text = "\n\n".join(
        f"{'=' * 18}\nPAGE {page['number']}\n{'=' * 18}\n\n"
        + page["text"]
        for page in pages
    )

    st.download_button(
        "📄 Download Complete Extracted PDF Text",
        data=full_text,
        file_name="Nestle_HR_Extracted_Text.txt",
        mime="text/plain",
        use_container_width=True
    )

    if st.session_state.chat_history:
        chat_export = "\n\n".join(
            f"{item['role'].upper()}:\n{item['content']}"
            for item in st.session_state.chat_history
        )

        st.download_button(
            "💬 Download Chat History",
            data=chat_export,
            file_name="Nestle_HR_Chat_History.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.caption("Ask a question to enable chat-history export.")

    st.write("All downloaded text is derived from your uploaded PDF.")


# ==========================================
# FOOTER
# ==========================================

st.divider()
st.caption(
    "💼 Nestlé HR Intelligence Assistant | "
    "Python • Streamlit • TF-IDF • PDF Processing"
)
