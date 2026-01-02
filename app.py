import streamlit as st
import os
import warnings
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

# -------------------- ENV --------------------
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not set in Hugging Face Secrets")
    st.stop()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_directory = os.path.join(os.path.dirname(__file__), "data")
vector_store_path = os.path.join(data_directory, "faiss_index")

# -------------------- EMBEDDINGS --------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 8}
    )

# -------------------- VECTOR STORE --------------------
@st.cache_resource
def load_vector_store():
    embedding_model = load_embeddings()
    return FAISS.load_local(
        vector_store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

if not os.path.exists(os.path.join(vector_store_path, "index.faiss")):
    st.error("⚠️ Vector database not found! Run vector_embedding.py first.")
    st.stop()

vector_store = load_vector_store()

# -------------------- FREE LLM (GROQ) --------------------
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=512,
)


# -------------------- PROMPT --------------------
prompt_template = """
As a highly knowledgeable fashion assistant, your role is to accurately interpret fashion queries and 
provide responses using our specialized fashion database. Follow these directives to ensure optimal user interactions:
1. Precision in Answers: Respond solely with information directly relevant to the user's query from our fashion database. 
    Refrain from making assumptions or adding extraneous details.
2. Topic Relevance: Limit your expertise to specific fashion-related areas:
    - Fashion Trends
    - Personal Styling Advice
    - Seasonal Wardrobe Selections
    - Accessory Recommendations
3. Handling Off-topic Queries: For questions unrelated to fashion (e.g., general knowledge questions like "Why is the sky blue?"), 
    politely inform the user that the query is outside the chatbot’s scope and suggest redirecting to fashion-related inquiries.
4. Promoting Fashion Awareness: Craft responses that emphasize good fashion sense, aligning with the latest trends and 
    personalized style recommendations.
5. Contextual Accuracy: Ensure responses are directly related to the fashion query, utilizing only pertinent 
    information from our database.
6. Relevance Check: If a query does not align with our fashion database, guide the user to refine their 
    question or politely decline to provide an answer.
7. Avoiding Duplication: Ensure no response is repeated within the same interaction, maintaining uniqueness and 
    relevance to each user query.
8. Streamlined Communication: Eliminate any unnecessary comments or closing remarks from responses. Focus on
    delivering clear, concise, and direct answers.
9. Avoid Non-essential Sign-offs: Do not include any sign-offs like "Best regards" or "FashionBot" in responses.
10. One-time Use Phrases: Avoid using the same phrases multiple times within the same response. Each 
    sentence should be unique and contribute to the overall message without redundancy.
Fashion Query:
{context}
Question: {question}
Answer:
"""

custom_prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -------------------- RAG --------------------
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

def get_response(question):
    response = rag_chain.invoke(question)
    return response.strip()

# -------------------- UI STYLES --------------------
st.markdown(
    """
    <style>
    .appview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<h3 style='text-align:left; border-bottom:3px solid red;'>
Discover the AI Styling Recommendations 👗👠
</h3>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("🤖 FashionBot")
    st.markdown("""
Hi! 👋 I'm here to help you with fashion choices.
**You can ask about:**
- Fashion Trends 👕
- Personal Styling 👢
- Seasonal Outfits 🌞
- Accessories 💍
""")

# -------------------- INITIAL MESSAGE --------------------
initial_message = """
Hi there! 👋 I'm your **FashionBot** 🤖  
Try asking:
- 🎀 What are the top fashion trends this summer?
- 🎀 Suggest an outfit for a summer wedding
- 🎀 Must-have winter accessories
- 🎀 Shoes for a cocktail dress
- 🎀 Best look for a professional photoshoot
"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": initial_message}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- CLEAR CHAT --------------------
def clear_chat():
    st.session_state.messages = [
        {"role": "assistant", "content": initial_message}
    ]

st.button("🧹 Clear Chat", on_click=clear_chat)

# -------------------- CHAT INPUT --------------------
if prompt := st.chat_input("Ask a fashion question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Fetching fashion advice..."):
            response = get_response(prompt)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
