import os
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
        with st.expander("📄 View Retrieved Source Chunks"):
            for match in top_hits:
                st.markdown(f"### **[{match['pdf_name']}]**")
                st.caption(f"**Rank:** {match['rank']} | **Page:** {match['page']} | **Lines:** {match['lines']}")
                st.info(match['text'])

                eval_results = evaluate_response(user_query, match['text'])
                if "error" in eval_results:
                    st.caption(f"⚠️ *Evaluation skipped for this chunk: {eval_results['error']}*")
                else:
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("ROUGE-1", f"{eval_results['rouge1']:.4f}")
                    m2.metric("ROUGE-2", f"{eval_results['rouge2']:.4f}")
                    m3.metric("ROUGE-L", f"{eval_results['rougeL']:.4f}")
                    m4.metric("BERT-P",  f"{eval_results['bert_precision']:.4f}")
                    m5.metric("BERT-R",  f"{eval_results['bert_recall']:.4f}")
                    m6.metric("BERT-F1", f"{eval_results['bert_f1']:.4f}")
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
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("ROUGE-1", f"{gen_eval_results['rouge1']:.4f}")
                    m2.metric("ROUGE-2", f"{gen_eval_results['rouge2']:.4f}")
                    m3.metric("ROUGE-L", f"{gen_eval_results['rougeL']:.4f}")
                    m4.metric("BERT-P",  f"{gen_eval_results['bert_precision']:.4f}")
                    m5.metric("BERT-R",  f"{gen_eval_results['bert_recall']:.4f}")
                    m6.metric("BERT-F1", f"{gen_eval_results['bert_f1']:.4f}")

            except Exception as e:
                st.error(f"An error occurred while running LangChain: {e}")