    st.session_state.doc_id = None

with st.sidebar:
    st.header("📁 Document workspace")
    uploaded = st.file_uploader("Upload a text-based HR policy PDF", type=["pdf"])
    top_k = st.slider("Relevant passages per answer", 1, 5, 3)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat = []
        st.rerun()
    st.info("Local PDF extraction + TF-IDF retrieval. No paid API key needed.")

if uploaded is None:
    st.info("👈 Upload your Nestlé HR policy PDF to activate the dashboard.")
    st.markdown("### Available tools\nPDF reader · Extractive summary · Search · Evidence-based chat · Text exports")
    st.stop()

try:
    raw = uploaded.getvalue()
    doc_hash = hashlib.sha256(raw).hexdigest()
    if st.session_state.doc_id != doc_hash:
        st.session_state.doc_id = doc_hash
        st.session_state.chat = []
    with st.spinner("Reading PDF..."):
        pages, passages = parse_pdf(raw)
    if not passages:
        st.warning("This PDF has no readable text passages. Scanned/image PDFs require OCR.")
        st.stop()
    vectorizer, matrix = build_index(doc_hash, tuple(p["text"] for p in passages))
except Exception as error:
    st.error(f"Could not read or index this PDF: {error}")
    st.stop()

st.success(f"Document loaded: {uploaded.name}")
a, b, c, d = st.columns(4)
a.metric("📄 Pages", len(pages))
b.metric("📝 Words", f"{sum(p['words'] for p in pages):,}")
c.metric("📚 Passages", len(passages))
d.metric("💬 Questions", sum(m["role"] == "user" for m in st.session_state.chat))

chat_tab, reader_tab, summary_tab, search_tab, export_tab = st.tabs(["💬 Ask", "📖 Read PDF", "🧠 Summarize", "🔎 Search", "📥 Download"])

with chat_tab:
    st.subheader("Ask about the policy")
    st.caption("Answers show the closest matching original text, not invented HR rules or GPT-generated claims.")
    examples = ["What are the HR principles?", "What does the policy say about employees?", "How are people developed?", "What is the purpose of this policy?"]
    cols = st.columns(2)
    chosen_question = None
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button("💡 " + example, key=f"example_{i}", use_container_width=True):
                chosen_question = example
    for item in st.session_state.chat:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("View full matched passages"):
                    for source in item["sources"]:
                        st.markdown(f"**PDF page {source['page']}**")
                        st.write(source["text"])
    entered_question = st.chat_input("Ask a question about the uploaded policy...")
    question = entered_question or chosen_question
    if question and question.strip():
        matches = search(question, passages, vectorizer, matrix, top_k)
        if matches:
            answer = f"**Best matching policy passage (page {matches[0]['page']}):**\n\n{matches[0]['text']}"
            if len(matches) > 1:
                answer += "\n\n**Additional relevant pages:** " + ", ".join(str(n) for n in sorted(set(m["page"] for m in matches[1:])))
            answer += "\n\n_Extracted text; verify the policy document for authoritative interpretation._"
        else:
            answer = "I could not find a relevant passage. Try different keywords or browse the PDF reader."
        st.session_state.chat.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer, "sources": matches}])
        st.rerun()

with reader_tab:
    st.subheader("Clean page-by-page extracted text")
    page_num = st.selectbox("Select page", [p["number"] for p in pages])
    selected = pages[page_num - 1]
    st.caption(f"{selected['words']} words · Page {page_num}")
    st.text_area("Extracted PDF text", value=selected["text"], height=400, key=f"read_{doc_hash}_{page_num}")
    st.download_button("Download selected page", selected["text"], f"nestle_hr_page_{page_num}.txt", "text/plain")
    with st.expander("Show document page previews"):
        for p in pages:
            st.markdown(f"**Page {p['number']}**")
            st.write(p["text"][:1200] or "No extractable text")

with summary_tab:
    st.subheader("Extractive policy summary")
    st.caption("Selects important original sentences; it is not an abstractive AI-generated summary.")
    length = st.radio("Summary length", ["Short", "Medium", "Detailed"], horizontal=True)
    summary = summarize(pages, {"Short": 4, "Medium": 8, "Detailed": 14}[length])
    summary_text = "\n\n".join(f"Page {x['page']}: {x['text']}" for x in summary)
    if summary:
        for s in summary:
            st.markdown(f"- **Page {s['page']}:** {s['text']}")
        st.download_button("Download summary", summary_text, "nestle_hr_summary.txt", "text/plain")
    else:
        st.info("Not enough readable sentences for a summary.")

with search_tab:
    st.subheader("Search across the full PDF")
    keyword = st.text_input("Enter keywords or a full question")
    if keyword.strip():
        hits = search(keyword, passages, vectorizer, matrix, 10)
        if not hits:
            st.warning("No closely matching passages found.")
        for i, hit in enumerate(hits, 1):
            with st.expander(f"Match {i} · Page {hit['page']} · Similarity {hit['score']:.0%}", expanded=i == 1):
                st.write(hit["text"])
                st.caption("Similarity is a retrieval score, not answer correctness.")

with export_tab:
    st.subheader("Download extracted information")
    all_text = "\n\n".join(f"{'='*12} PAGE {p['number']} {'='*12}\n{p['text']}" for p in pages)
    st.download_button("📄 Download all extracted PDF text", all_text, "nestle_hr_full_text.txt", "text/plain", use_container_width=True)
    if st.session_state.chat:
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat)
        st.download_button("💬 Download conversation", transcript, "nestle_hr_conversation.txt", "text/plain", use_container_width=True)
    else:
        st.caption("Your conversation download will appear after you ask a question.")

st.divider()
st.caption("Nestlé HR Policy Explorer · Python · Streamlit · PDF extraction · TF-IDF · No paid API")
