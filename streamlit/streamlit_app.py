
import re
import math
import hashlib
from io import BytesIO
from collections import Counter

import streamlit as st
from pypdf import PdfReader

# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Nestlé HR Intelligence Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# COLORFUL PROFESSIONAL DESIGN
# =====================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        120deg,
        #f3efff 0%,
        #eaf6ff 65%,
        #fff1f8 100%
    );
}

[data-testid="stSidebar"] {
    background: #eeebff;
}

.hero {
    padding: 35px;
    border-radius: 22px;
    background: linear-gradient(
        110deg,
        #4338ca,
        #7c3aed,
        #c21869
    );
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 28px rgba(80,60,170,0.20);
}

.hero h1,
.hero p {
    color: white !important;
}

.hero h1 {
    font-size: 2.3rem;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid #ddd6fe;
    border-radius: 15px;
    padding: 15px;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.90);
    border: 1px solid #ddd6fe;
    border-radius: 15px;
    padding: 12px;
}

.stButton button {
    border-radius: 12px;
    font-weight: 600;
}

.stDownloadButton button {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💼 Nestlé HR Intelligence Assistant</h1>
    <p>
        Your Smart Human Resources Document Assistant
    </p>
    <p>
        📄 PDF Extraction &nbsp; | &nbsp;
        🖼️ OCR Scanner &nbsp; | &nbsp;
        💬 HR Chat &nbsp; | &nbsp;
        📝 Summaries &nbsp; | &nbsp;
        🔎 Policy Search
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================
# TEXT PROCESSING
# =====================================

STOPWORDS = set("""
the a an and or to of for in on is are was were
be been by with as at it its that this these those
from do does how what who why which about into
your you they their our can could would should
""".split())


def tokenize(text):
    return [
        word
        for word in re.findall(
            r"[a-z]+",
            text.lower()
        )
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


def split_into_passages(text, size=650, overlap=110):
    if not text:
        return []

    results = []
    start = 0

    while start < len(text):
        end = min(len(text), start + size)

        if end < len(text):
            pivot = text.rfind(
                " ",
                start + size // 2,
                end
            )

            if pivot > start:
                end = pivot

        passage = text[start:end].strip()

        if len(passage) > 20:
            results.append(passage)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return results


# =====================================
# PDF READING + OCR
# =====================================

@st.cache_data(show_spinner=False, max_entries=4)
def extract_pdf(pdf_bytes, use_ocr):

    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []
    pdf_document = None

    for index, page in enumerate(reader.pages):

        text = clean_text(
            page.extract_text() or ""
        )

        method = "Original PDF text"

        # OCR fallback for scanned pages
        if len(text) < 40 and use_ocr:

            try:
                import fitz
                import pytesseract
                from PIL import Image

                if pdf_document is None:
                    pdf_document = fitz.open(
                        stream=pdf_bytes,
                        filetype="pdf"
                    )

                pdf_page = pdf_document[index]

                pix = pdf_page.get_pixmap(
                    matrix=fitz.Matrix(1.7, 1.7),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                ocr_text = clean_text(
                    pytesseract.image_to_string(
                        image,
                        lang="eng"
                    )
                )

                if len(ocr_text) > len(text):
                    text = ocr_text
                    method = "OCR scanned text"

            except Exception as error:
                method = (
                    "OCR unavailable: "
                    + str(error)[:100]
                )

        pages.append({
            "page": index + 1,
            "text": text,
            "method": method
        })

    if pdf_document is not None:
        pdf_document.close()

    return pages


# =====================================
# BUILD SEARCHABLE PASSAGES
# =====================================

def build_passages(pages):

    passages = []

    for page in pages:

        sections = split_into_passages(
            page["text"]
        )

        for section in sections:
            passages.append({
                "page": page["page"],
                "text": section
            })

    return passages


# =====================================
# INTELLIGENT DOCUMENT SEARCH
# =====================================

def search_document(
    question,
    passages,
    limit=4
):

    query_words = Counter(
        tokenize(question)
    )

    if not query_words:
        return []

    documents = [
        Counter(tokenize(p["text"]))
        for p in passages
    ]

    total = len(documents)

    if total == 0:
        return []

    document_frequency = Counter()

    for doc in documents:
        document_frequency.update(
            doc.keys()
        )

    average_length = (
        sum(sum(doc.values()) for doc in documents)
        / total
    ) or 1

    scored = []

    for index, doc in enumerate(documents):

        document_length = sum(doc.values())
        score = 0.0

        for term in query_words:

            frequency = doc.get(term, 0)

            if not frequency:
                continue

            df = document_frequency[term]

            idf = math.log(
                1 + (
                    total - df + 0.5
                ) / (df + 0.5)
            )

            score += (
                idf * frequency * 2.2
                / (
                    frequency
                    + 1.2 * (
                        0.25
                        + 0.75
                        * document_length
                        / average_length
                    )
                )
            )

        if score > 0:
            scored.append((score, index))

    scored.sort(reverse=True)

    results = []

    for score, index in scored[:limit]:

        results.append({
            "page": passages[index]["page"],
            "text": passages[index]["text"],
            "score": round(score, 2)
        })

    return results


# =====================================
# EXTRACTIVE DOCUMENT SUMMARY
# =====================================

def generate_summary(pages, limit=8):

    candidates = []

    for page in pages:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            page["text"]
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if 45 <= len(sentence) <= 350:

                candidates.append({
                    "page": page["page"],
                    "text": sentence
                })

    if not candidates:

        return [
            {
                "page": page["page"],
                "text": page["text"][:240]
            }
            for page in pages
            if page["text"]
        ][:limit]

    frequencies = Counter(
        word
        for candidate in candidates
        for word in tokenize(candidate["text"])
    )

    ranked = []

    for index, candidate in enumerate(candidates):

        words = tokenize(
            candidate["text"]
        )

        score = (
            sum(frequencies[word] for word in words)
            / math.sqrt(max(1, len(words)))
        )

        ranked.append((score, index))

    ranked.sort(reverse=True)

    selected = sorted(
        index
        for score, index in ranked[:limit]
    )

    return [
        candidates[index]
        for index in selected
    ]


# =====================================
# SESSION MEMORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_id" not in st.session_state:
    st.session_state.document_id = None


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("📁 HR Document Center")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"]
    )

    use_ocr = st.toggle(
        "🖼️ Read scanned pages with OCR",
        value=True
    )

    result_limit = st.slider(
        "Matching passages per question",
        min_value=1,
        max_value=6,
        value=3
    )

    if st.button(
        "🗑️ Clear Chat History",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("✨ Features")

    st.write("✅ Normal PDF extraction")
    st.write("✅ Scanned PDF OCR")
    st.write("✅ HR document chat")
    st.write("✅ PDF reader")
    st.write("✅ Automatic summary")
    st.write("✅ Policy keyword search")
    st.write("✅ Downloadable results")

    st.info(
        "Scanned PDFs may take longer to process."
    )


# =====================================
# INITIAL SCREEN
# =====================================

if uploaded_file is None:

    st.info(
        "👈 Upload an HR Policy PDF "
        "from the sidebar to begin."
    )

    st.subheader(
        "🚀 Your HR Document Intelligence Workspace"
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 📄 Extract")
        st.write(
            "Read text from standard and scanned PDFs."
        )

    with b:
        st.markdown("### 📝 Summarize")
        st.write(
            "Identify important sentences from the document."
        )

    with c:
        st.markdown("### 💬 Ask")
        st.write(
            "Find policy information by asking questions."
        )

    st.stop()


# =====================================
# DOCUMENT PROCESSING
# =====================================

pdf_bytes = uploaded_file.getvalue()

document_hash = hashlib.sha256(
    pdf_bytes
).hexdigest()

if st.session_state.document_id != document_hash:

    st.session_state.document_id = document_hash
    st.session_state.messages = []

try:

    with st.spinner(
        "📄 Extracting PDF text and processing scanned pages..."
    ):

        pages = extract_pdf(
            pdf_bytes,
            use_ocr
        )

        passages = build_passages(pages)

except Exception as error:

    st.error(
        "Could not open or process this PDF."
    )

    st.caption(str(error))
    st.stop()


# =====================================
# DOCUMENT DASHBOARD
# =====================================

readable_pages = sum(
    bool(page["text"])
    for page in pages
)

ocr_pages = sum(
    page["method"] == "OCR scanned text"
    for page in pages
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("📄 Total Pages", len(pages))
c2.metric("🔤 Readable Pages", readable_pages)
c3.metric("🖼️ OCR Pages", ocr_pages)
c4.metric("🧩 Search Sections", len(passages))

if not passages:

    st.error(
        "No readable text was extracted. "
        "Please enable OCR for scanned PDFs."
    )

    with st.expander("View extraction details"):
        for page in pages:
            st.write(
                f"Page {page['page']}: {page['method']}"
            )

    st.stop()

st.success(
    f"✅ Document ready: {uploaded_file.name}"
)

st.divider()


# =====================================
# MAIN TABS
# =====================================

chat_tab, reader_tab, summary_tab, search_tab, download_tab = st.tabs([
    "💬 HR Chat",
    "📖 PDF Reader",
    "📝 Summary",
    "🔎 Search",
    "📥 Downloads"
])


# =====================================
# TAB 1: HR CHAT
# =====================================

with chat_tab:

    st.subheader("💬 Ask Your HR Assistant")

    st.caption(
        "Ask a question and receive relevant "
        "policy passages with page references."
    )

    suggestions = [
        "What are the main HR principles?",
        "What does the policy say about employees?",
        "How does Nestlé support employee development?",
        "What are the responsibilities of employees?"
    ]

    selected_question = st.selectbox(
        "Quick Questions",
        [""] + suggestions
    )

    ask_selected = st.button(
        "✨ Ask Selected Question"
    )

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["text"]
            )

    typed_question = st.chat_input(
        "Ask anything about the uploaded policy..."
    )

    question = (
        typed_question
        or (
            selected_question
            if ask_selected else ""
        )
    )

    if question:

        results = search_document(
            question,
            passages,
            result_limit
        )

        if results:

            best = results[0]

            answer = (
                f"### 📚 Relevant HR Policy Information\n\n"
                f"**Source Page {best['page']}**\n\n"
                f"{best['text']}\n\n"
                f"**Other matching pages:** "
                + ", ".join(
                    str(page)
                    for page in sorted({
                        result["page"]
                        for result in results
                    })
                )
            )

        else:

            answer = (
                "I couldn't find a matching policy passage. "
                "Try another question or different keywords."
            )

        st.session_state.messages.extend([
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


# =====================================
# TAB 2: PDF READER
# =====================================

with reader_tab:

    st.subheader("📖 PDF Text Reader")

    page_number = st.selectbox(
        "Select PDF Page",
        range(1, len(pages) + 1)
    )

    selected_page = pages[
        page_number - 1
    ]

    st.caption(
        "Extraction Method: "
        + selected_page["method"]
    )

    st.text_area(
        "Clean Extracted Text",
        value=selected_page["text"],
        height=450
    )

    st.download_button(
        "📥 Download This Page",
        data=selected_page["text"],
        file_name=f"HR_Page_{page_number}.txt",
        mime="text/plain"
    )


# =====================================
# TAB 3: DOCUMENT SUMMARY
# =====================================

with summary_tab:

    st.subheader(
        "📝 Automatic Document Summary"
    )

    summary_length = st.radio(
        "Choose Summary Length",
        [
            "Short",
            "Medium",
            "Detailed"
        ],
        horizontal=True
    )

    limit = {
        "Short": 4,
        "Medium": 8,
        "Detailed": 14
    }[summary_length]

    summary_items = generate_summary(
        pages,
        limit
    )

    summary_text = "\n\n".join(
        f"Page {item['page']}: {item['text']}"
        for item in summary_items
    )

    if summary_items:

        for item in summary_items:

            st.markdown(
                f"**Page {item['page']}:** "
                f"{item['text']}"
            )

        st.download_button(
            "📥 Download Summary",
            data=summary_text,
            file_name="Nestle_HR_Summary.txt",
            mime="text/plain"
        )

    else:
        st.warning(
            "No suitable text found for summarization."
        )

    st.caption(
        "The summary selects important original "
        "sentences from the uploaded PDF."
    )


# =====================================
# TAB 4: POLICY SEARCH
# =====================================

with search_tab:

    st.subheader("🔎 Search HR Policies")

    keyword = st.text_input(
        "Enter a keyword, topic, or question",
        placeholder="Example: employee development"
    )

    if keyword:

        matches = search_document(
            keyword,
            passages,
            10
        )

        if not matches:
            st.warning(
                "No relevant passages found."
            )

        for index, item in enumerate(
            matches,
            start=1
        ):

            with st.expander(
                f"Result {index} — Page {item['page']}",
                expanded=(index == 1)
            ):
                st.write(
                    item["text"]
                )


# =====================================
# TAB 5: DOWNLOAD CENTER
# =====================================

with download_tab:

    st.subheader("📥 Download Center")

    full_text = "\n\n".join(
        f"========== PAGE {page['page']} ==========\n"
        + page["text"]
        for page in pages
    )

    st.download_button(
        "📄 Download Complete PDF Text",
        data=full_text,
        file_name="Nestle_HR_Extracted_Text.txt",
        mime="text/plain",
        use_container_width=True
    )

    if st.session_state.messages:

        chat_text = "\n\n".join(
            f"{item['role'].upper()}:\n"
            f"{item['text']}"
            for item in st.session_state.messages
        )

        st.download_button(
            "💬 Download Chat History",
            data=chat_text,
            file_name="Nestle_HR_Conversation.txt",
            mime="text/plain",
            use_container_width=True
        )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "💼 Nestlé HR Intelligence Assistant | "
    "Python • Streamlit • PDF Extraction • OCR"
)
