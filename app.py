import os
import re
import streamlit as st
import dotenv

# Import LangChain core utilities
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Import the search function from your existing script
from src.cosine_similarity import search
from src.create_embedding import run_embeddings
from src.evaluation import evaluate_response
from src.prompt import RAG_prompt

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(page_title="PDF Semantic Search", page_icon="🔍")
st.title("🔍 PDF Document Semantic Search (LangChain)")
st.write("Ask a question to search through your processed PDFs via LangChain chains.")

# Verify API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("❌ API key is missing. Please check your .env file.")
    st.stop()

llm = ChatOpenAI(model="gpt-5.5", temperature=0)

FOLDER_PATH = "data/"
os.makedirs(FOLDER_PATH, exist_ok=True)


@st.cache_data(show_spinner=False)
def render_pdf_page_to_image(pdf_path: str, page_number: int, zoom: float = 3.0) -> bytes:
    import fitz

    page_number = int(page_number)
    if page_number < 1:
        raise ValueError("Page number must be 1 or greater.")

    with fitz.open(pdf_path) as doc:
        if page_number > doc.page_count:
            raise ValueError(f"Page {page_number} is outside the PDF page range of 1-{doc.page_count}.")

        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")


def generated_answer_cites_more_than_top_match(generated_answer: str, top_match) -> bool:
    cited_pages = set(re.findall(r"\bPage\s+(\d+)\b", generated_answer, flags=re.IGNORECASE))
    cited_lines = {
        line_range.replace(" ", "")
        for line_range in re.findall(
            r"\bLines?\s+(\d+(?:\s*-\s*\d+)?)\b",
            generated_answer,
            flags=re.IGNORECASE,
        )
    }

    top_page = str(top_match.get("page", "")).strip()
    top_lines = str(top_match.get("lines", "")).replace(" ", "")

    return any(page != top_page for page in cited_pages) or any(lines != top_lines for lines in cited_lines)


def show_pdf_page_image(image_bytes: bytes, caption: str):
    try:
        st.image(image_bytes, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image_bytes, caption=caption, use_column_width=True)


def show_evidence_check(top_match, generated_answer: str):
    pdf_name = top_match.get("pdf_name", "Unknown PDF")
    page_value = top_match.get("page")
    lines_value = top_match.get("lines", "Unknown")
    rank_value = top_match.get("rank", "Unknown")
    distance_value = top_match.get("distance_score")
    chunk_text = top_match.get("text", "")
    distance_label = (
        f"{distance_value:.4f}" if isinstance(distance_value, (int, float)) else "unavailable"
    )

    st.subheader("Evidence Check: Top Retrieved Result vs PDF Page")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("#### Top Retrieved Chunk")
        st.markdown(f"**PDF:** {pdf_name}")
        st.markdown(f"**Page:** {page_value if page_value is not None else 'Missing'}")
        st.markdown(f"**Extracted text lines:** {lines_value}")
        st.markdown(f"**Rank:** {rank_value}")
        st.caption(f"**L2 distance:** {distance_label} — lower is better")
        st.info(chunk_text)

    with right_col:
        st.markdown("#### Rendered Source Page")
        pdf_path = os.path.join(FOLDER_PATH, pdf_name)
        caption = f"Rendered source page: {pdf_name}, page {page_value if page_value is not None else 'missing'}"
        st.caption(caption)
        st.caption("This is the original PDF page corresponding to the top retrieved chunk.")

        if not os.path.exists(pdf_path):
            st.warning(f"Source PDF not found: {pdf_path}")
            return

        if page_value is None:
            st.warning("Missing page number in retrieved metadata.")
            return

        try:
            page_number = int(page_value)
        except (TypeError, ValueError):
            st.warning(f"Invalid page number in retrieved metadata: {page_value}")
            return

        try:
            image_bytes = render_pdf_page_to_image(pdf_path, page_number)
            show_pdf_page_image(image_bytes, caption)
        except Exception as e:
            st.warning(f"Could not render source PDF page: {e}")

    if generated_answer_cites_more_than_top_match(generated_answer, top_match):
        st.info(
            "The generated answer appears to cite additional evidence not shown in the top-1 comparison. "
            "The current panel verifies only the top retrieved chunk. Check the retrieved chunks expander "
            "for the remaining cited evidence."
        )


# -- Side bar to upload document--
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader("Upload a new PDF document to data folder", type=["pdf"])
    
    if uploaded_file is not None:
        file_path = os.path.join(FOLDER_PATH, uploaded_file.name)
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}")
        else:
            st.info(f"ℹ️ {uploaded_file.name} is already in the data folder.")

    st.markdown("---")
    st.subheader("⚙️ Index Operations")
    
    # The button triggers your backend script logic directly
    if st.button("🔄 Rebuild Vector Database", type="primary"):
        with st.spinner("Processing documents and generating embeddings..."):
            status_message = run_embeddings()
            
            if status_message and "Success" in status_message:
                st.success(status_message)
            elif status_message:
                st.error(status_message)
            else:
                st.error("❌ An unknown error occurred during indexing.")

# --- UI Interface ---
user_query = st.text_input(
    "Enter your query:", 
    placeholder="e.g., What problem does the Transformer solve compared to RNNs?"
)

num_results = st.slider("Number of results to return from RAG:", min_value=1, max_value=5, value=2)

if user_query:
    # 1. Step 1: Search and retrieve context chunks using your existing script
    with st.spinner("Searching document embeddings..."):
        try:
            top_hits = search(user_query, k=num_results)
        except Exception as e:
            st.error(f"An error occurred during search: {e}")
            top_hits = []

    if not top_hits:
        st.info("No matching chunks found in the database.")
    else:
        # Display the source documents in an expander
        with st.expander("📄 View all retrieved chunks used as RAG context"):
            for match in top_hits:
                st.markdown(f"### **[{match['pdf_name']}]**")
                st.caption(
                    f"**Rank:** {match['rank']} | "
                    f"**Page:** {match['page']} | "
                    f"**Extracted text lines:** {match['lines']} | "
                    f"**L2 distance:** {match['distance_score']:.4f} — lower is better"
                )
                st.info(match['text'])

                eval_results = evaluate_response(user_query, match['text'])
                if "error" in eval_results:
                    st.caption(f"⚠️ *Evaluation skipped for this chunk: {eval_results['error']}*")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ROUGE-1", f"{eval_results['rouge1']:.4f}")
                    m2.metric("ROUGE-2", f"{eval_results['rouge2']:.4f}")
                    m3.metric("ROUGE-L", f"{eval_results['rougeL']: .4f}")
                st.markdown("---")

        # 2. Step 2: Build the context string from RAG outputs
        context_block = ""
        for match in top_hits:
            context_block += f"--- START CHUNK FROM {match['pdf_name']} (Page {match['page']}) (Line {match['lines']}) ---\n"
            context_block += f"{match['text']}\n"
            context_block += "--- END CHUNK ---\n\n"

        # 3. Step 3: LangChain Pipeline Setup
        with st.spinner("Synthesizing answer with LangChain..."):
            try:
                system = RAG_prompt.format(context=context_block, question=user_query)
                message = [SystemMessage(content=system), HumanMessage(content="Please answer the question")]
                
                # Combine into an LCEL chain (Prompt -> Model -> String Parser)
                rag_chain = llm | StrOutputParser()
                
                # Invoke the chain, passing the exact key inputs needed by the template
                generated_answer = rag_chain.invoke(message)
                
                # Output the synthesized answer
                st.subheader("🤖 Generated Answer")
                st.write(generated_answer)
                gen_eval_results = evaluate_response(user_query, generated_answer)
                if "error" in gen_eval_results:
                    st.caption(f"⚠️ *Evaluation skipped for this chunk: {gen_eval_results['error']}*")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("ROUGE-1", f"{gen_eval_results['rouge1']:.4f}")
                    m2.metric("ROUGE-2", f"{gen_eval_results['rouge2']:.4f}")
                    m3.metric("ROUGE-L", f"{gen_eval_results['rougeL']: .4f}")


            except Exception as e:
                st.error(f"An error occurred while running LangChain: {e}")
