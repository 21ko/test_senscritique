import streamlit as st
import pandas as pd
from sentence_transformers import util
from cleanig import Clean 

cleaner = Clean()
model = cleaner.model  
def find_similar_comment(user_input, df, embeddings, top_k=3):
    user_clean = cleaner.clean_text(user_input)
    user_emb = model.encode(user_clean, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_emb, embeddings)[0]
    top_results = cosine_scores.topk(k=top_k)

    results = []
    for s, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        results.append((df["review_content"].iloc[idx], float(s)))

    return results



st.title("🎬 Comment Similarity Finder")

st.write("Upload a CSV or choose one of the raw movie datasets.")


source = st.radio("Select data source:", ["Upload CSV", "Use existing raw movie files"])

df = None

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

else:
    file_choice = st.selectbox(
        "Choose an existing RAW file:",
        ["fightclub_critiques.csv", "interstellar_critique.csv"]  
    )
    df = pd.read_csv(file_choice)



if df is not None:

    
   
   
    st.info("Cleaning comments…")
    df["comment_clean"] = df["review_content"].astype(str).apply(cleaner.clean_text)
    st.success("✨ Cleaning complete!")

    
    st.info("Computing embeddings… (first time may take a bit)")
    comments = df["comment_clean"].tolist()
    embeddings = model.encode(comments, convert_to_tensor=True)
    st.success("🎉 Embeddings ready!")

    # ==========================
    # 5. User Query Section
    # ==========================
    st.header("💬 Search for Similar Comments")

    mode = st.radio("Input mode:", ["Type a comment", "Select from dataset"])

    if mode == "Type a comment":
        user_comment = st.text_area("Enter your comment:")
    else:
        user_comment = st.selectbox("Pick a comment from the dataset:",
                                    df["review_content"].tolist())

    if st.button("🔍 Find Similar Comments"):
        if user_comment.strip() == "":
            st.error("⚠️ Please enter a comment.")
        else:
            results = find_similar_comment(user_comment, df, embeddings)

            st.subheader("📊 Most Similar Comments")
            for text, score in results:
                st.markdown(f"**📝 Comment:** {text}")
                st.markdown(f"**🔥 Similarity:** `{score:.4f}`")
                st.markdown("---")
