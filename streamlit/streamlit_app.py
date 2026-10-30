
import streamlit as st
import re
import math
import hashlib
from io import BytesIO
from collections import Counter
from pypdf import PdfReader

st.set_page_config(
    page_title="Nestlé HR Assistant",
    page_icon="💼",
    layout="wide"
)

# ============================
# PROFESSIONAL DESIGN
# ============================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        120deg, #f4f0ff, #eaf5ff 65%, #fff1f9
    );
}
[data-testid="stSidebar"] {
    background: #f0edff;
}
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(
        110deg, #4338ca, #7c3aed, #be185d
    );
    color: white;
    margin-bottom: 24px;
}
.hero h1, .hero p {
    color: white !important;
}
[data-testid="stMetric"] {
    background: #ffffffdf;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #ddd6fe;
}
[data-testid="stChatMessage"] {
    background: #ffffffdf;
    border-radius: 15px;
    border: 1px solid #ddd6fe;
}
.stButton button {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>💼 Nestlé HR Intelligence Assistant</h1>
<p>Explore and understand Human Resources policies</p>
<p>📄 PDF Reader | 💬 Smart Chat | 📝 Summary |
🔎 Search | 📥 Downloads</p>
</div>
""", unsafe_allow_html=True)

# ============================
# TEXT PROCESSING
# ============================

STOPWORDS = set("""
the a an and or to for of in on is are was were
be been being by with as at it its that this
these those from do does did how what which who
when where why can could would should about into
than then there their you your we our they them
""".split())

def tokenize(text):
    words = re.findall(
        r"[a-zA-Z][a-zA-Z'-]*",
        text.lower()
    )
    return [
        word for word in words
        if len(word) > 2 and word not in STOPWORDS
    ]

def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text
        )
        if len(sentence.strip()) >= 35
    ]

# ============================
# PDF READER
# ============================

@st.cache_data(show_spinner=False)
def load_pdf(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    passages = []

    for number, page in enumerate(
        reader.pages, start=1
    ):
        text = clean_text(
            page.extract_text() or ""
        )

        pages.append({
            "number": number,
            "text": text,
            "words": len(text.split())
        })

        sentences = split_sentences(text)

        if not sentences and len(text) > 35:
            sentences = [
                text[i:i + 650]
                for i in range(0, len(text), 650)
            ]

        for i in range(0, len(sentences), 2):
            passage = " ".join(
                sentences[i:i + 2]
            ).strip()

            if passage:
                passages.append({
                    "page": number,
                    "text": passage
                })

    return pages, passages

# ============================
# SEARCH INDEX
# ============================

@st.cache_data(show_spinner=False)
def create_index(passages):
    documents = [
        Counter(tokenize(item["text"]))
        for item in passages
    ]

    document_frequency = Counter()

    for doc in documents:
        document_frequency.update(doc.keys())

    return documents, document_frequency

def search_policy(
    question,
    passages,
    documents,
    document_frequency,
    limit=3
):
    terms = set(tokenize(question))

    if not terms:
        return []

    total = len(documents)

    avg_length = (
        sum(sum(doc.values()) for doc in documents)
        / max(total, 1)
    )

    scores = []

    for index, doc in enumerate(documents):
        doc_length = sum(doc.values())
        score = 0.0

        for term in terms:
            frequency = doc.get(term, 0)

            if not frequency:
                continue

            df = document_frequency.get(term, 0)

            idf = math.log(
                1 + (
                    total - df + 0.5
                ) / (df + 0.5)
            )

            score += (
                idf * frequency * 2.2 /
                (
                    frequency +
                    1.2 * (
                        0.25 +
                        0.75 * doc_length /
                        max(avg_length, 1)
                    )
                )
            )

        if score > 0:
            scores.append((score, index))

    scores.sort(reverse=True)

    return [
        {
            "page": passages[index]["page"],
            "text": passages[index]["text"],
            "score": round(score, 2)
        }
        for score, index in scores[:limit]
    ]

# ============================
# AUTOMATIC SUMMARY
# ============================

def generate_summary(pages, limit):
    sentences = []

    for page in pages:
        for sentence in split_sentences(
            page["text"]
        ):
            if 45 <= len(sentence) <= 450:
                sentences.append({
                    "page": page["number"],
                    "text": sentence
                })

    if not sentences:
        return []

    frequencies = Counter(
        word
        for item in sentences
        for word in tokenize(item["text"])
    )

    ranked = []

    for index, item in enumerate(sentences):
        words = tokenize(item["text"])

        score = (
            sum(frequencies[word] for word in words)
            / math.sqrt(max(len(words), 1))
        )

        ranked.append((score, index))

    ranked.sort(reverse=True)

    selected = sorted(
        index for score, index in ranked[:limit]
    )

    return [sentences[index] for index in selected]

# ============================
# SESSION STATE
# ============================

if "chat" not in st.session_state:
    st.session_state.chat = []

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

# ============================
# SIDEBAR
# ============================

with st.sidebar:
    st.title("📁 HR Document Center")

    uploaded_pdf = st.file_uploader(
        "Upload an HR Policy PDF",
        type=["pdf"]
    )

    result_limit = st.slider(
        "Number of matching passages",
        1, 5, 3
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.chat = []
        st.rerun()

    st.divider()

    st.subheader("✨ Features")
    st.write("✅ Document analysis")
    st.write("✅ HR policy questions")
    st.write("✅ PDF text extraction")
    st.write("✅ Document summary")
    st.write("✅ Policy keyword search")
    st.write("✅ Downloadable results")

    st.caption(
        "Free educational document assistant."
    )

# ============================
# UPLOAD SCREEN
# ============================

if uploaded_pdf is None:
    st.info(
        "👈 Upload your Nestlé HR Policy PDF "
        "using the sidebar to begin."
    )
    st.stop()

# ============================
# PROCESS DOCUMENT
# ============================

try:
    file_bytes = uploaded_pdf.getvalue()

    current_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    if (
        st.session_state.document_hash
        != current_hash
    ):
        st.session_state.document_hash = current_hash
        st.session_state.chat = []

    with st.spinner("Processing PDF..."):
        pages, passages = load_pdf(file_bytes)

    if not passages:
        st.warning(
            "No readable text found. "
            "Scanned PDFs may require OCR."
        )
        st.stop()

    documents, document_frequency = create_index(
        passages
    )

except Exception as error:
    st.error("Unable to process the PDF.")
    st.caption(str(error))
    st.stop()

st.success(
    f"✅ PDF loaded successfully: {uploaded_pdf.name}"
)

# ============================
# DOCUMENT STATISTICS
# ============================

col1, col2, col3 = st.columns(3)

col1.metric("📄 PDF Pages", len(pages))
col2.metric(
    "📝 Total Words",
    sum(page["words"] for page in pages)
)
col3.metric(
    "🧩 Searchable Passages",
    len(passages)
)

st.divider()

# ============================
# APPLICATION TABS
# ============================

chat_tab, reader_tab, summary_tab, search_tab, download_tab = st.tabs([
    "💬 HR Chat",
    "📖 PDF Reader",
    "📝 Summary",
    "🔎 Search",
    "📥 Downloads"
])

# ============================
# CHAT TAB
# ============================

with chat_tab:
    st.subheader("💬 Ask Your HR Assistant")

    suggested_questions = [
        "What are the HR policy principles?",
        "What does the policy say about employees?",
        "How does Nestlé address training and development?"
    ]

    suggestion = st.selectbox(
        "Suggested questions",
        [""] + suggested_questions
    )

    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    question = st.chat_input(
        "Ask a question about the HR policy..."
    )

    if st.button("Ask Suggested Question"):
        if suggestion:
            question = suggestion

    if question:
        results = search_policy(
            question,
            passages,
            documents,
            document_frequency,
            result_limit
        )

        if results:
            best = results[0]

            answer = (
                f"**Relevant Policy Information "
                f"— Page {best['page']}**\n\n"
                f"{best['text']}\n\n"
                "**Source Pages:** "
                + ", ".join(
                    str(page)
                    for page in sorted({
                        item["page"]
                        for item in results
                    })
                )
            )
        else:
            answer = (
                "No matching passage found. "
                "Try a more specific question."
            )

        st.session_state.chat.extend([
            {
                "role": "user",
                "text": question
            },
            {
                "role": "assistant",
                "text": answer
            }
        ])

        st.rerun()

# ============================
# PDF READER TAB
# ============================

with reader_tab:
    st.subheader("📖 Clean PDF Text Reader")

    page_number = st.selectbox(
        "Choose PDF Page",
        [page["number"] for page in pages]
    )

    selected_page = pages[page_number - 1]

    st.text_area(
        "Extracted Text",
        value=selected_page["text"],
        height=400
    )

    st.download_button(
        "📥 Download This Page",
        data=selected_page["text"],
        file_name=f"HR_Page_{page_number}.txt"
    )

# ============================
# SUMMARY TAB
# ============================

with summary_tab:
    st.subheader("📝 Automatic Policy Summary")

    length = st.radio(
        "Choose Summary Length",
        ["Short", "Medium", "Detailed"],
        horizontal=True
    )

    limit = {
        "Short": 4,
        "Medium": 8,
        "Detailed": 14
    }[length]

    summary = generate_summary(
        pages,
        limit
    )

    summary_text = "\n\n".join(
        f"Page {item['page']}: {item['text']}"
        for item in summary
    )

    if summary:
        for item in summary:
            st.markdown(
                f"- **Page {item['page']}:** "
                f"{item['text']}"
            )

        st.download_button(
            "📥 Download Summary",
            data=summary_text,
            file_name="Nestle_HR_Summary.txt"
        )
    else:
        st.warning(
            "No suitable sentences found for summarization."
        )

    st.caption(
        "This is an extractive summary using "
        "sentences from the PDF."
    )

# ============================
# SEARCH TAB
# ============================

with search_tab:
    st.subheader("🔎 Search HR Policies")

    keyword = st.text_input(
        "Enter a keyword or topic",
        placeholder="Example: employee development"
    )

    if keyword:
        matches = search_policy(
            keyword,
            passages,
            documents,
            document_frequency,
            10
        )

        if not matches:
            st.warning("No relevant passages found.")

        for index, item in enumerate(matches, 1):
            with st.expander(
                f"Result {index} — Page {item['page']}",
                expanded=(index == 1)
            ):
                st.write(item["text"])

# ============================
# DOWNLOAD TAB
# ============================

with download_tab:
    st.subheader("📥 Download Center")

    all_text = "\n\n".join(
        f"========== PAGE {page['number']} ==========\n"
        + page["text"]
        for page in pages
    )

    st.download_button(
        "📄 Download Full Extracted PDF Text",
        data=all_text,
        file_name="Nestle_HR_Extracted_Text.txt",
        use_container_width=True
    )

    if st.session_state.chat:
        chat_text = "\n\n".join(
            f"{message['role'].upper()}:\n"
            f"{message['text']}"
            for message in st.session_state.chat
        )

        st.download_button(
            "💬 Download Chat History",
            data=chat_text,
            file_name="Nestle_HR_Chat.txt",
            use_container_width=True
        )

# ============================
# FOOTER
# ============================

st.divider()

st.caption(
    "💼 Nestlé HR Intelligence Assistant | "
    "Python • Streamlit • PDF Processing"
)
