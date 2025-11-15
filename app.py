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


st.title("Recherche de Commentaires Similaires")

st.write("Importez un fichier CSV ou choisissez l’un des jeux de données bruts disponibles.")


source = st.radio("Sélectionnez la source des données :", ["Importer un CSV", "Utiliser un fichier existant"])

df = None

if source == "Importer un CSV":
    uploaded = st.file_uploader("Importer votre fichier CSV :", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

else:
    file_choice = st.selectbox(
        "Choisissez un fichier brut existant :",
        ["fightclub_critiques.csv", "interstellar_critique.csv"]
    )
    df = pd.read_csv(file_choice)



if df is not None:

    
    df["comment_clean"] = df["review_content"].astype(str).apply(cleaner.clean_text)
    

    
    
    comments = df["comment_clean"].tolist()
    embeddings = model.encode(comments, convert_to_tensor=True)
   

    mode = st.radio("Mode d’entrée :", ["Saisir un commentaire", "Sélectionner un commentaire existant"])

    if mode == "Saisir un commentaire":
        user_comment = st.text_area("Entrez votre commentaire :")
    else:
        user_comment = st.selectbox("Choisissez un commentaire dans le dataset :",
                                    df["review_content"].tolist())

    if st.button("Rechercher les commentaires similaires"):
        if user_comment.strip() == "":
            st.error("Veuillez saisir un commentaire.")
        else:
            results = find_similar_comment(user_comment, df, embeddings)

            st.subheader("Commentaires les plus similaires")
            for text, score in results:
                st.markdown(f"**Commentaire :** {text}")
                st.markdown(f"**Score de similarité :** `{score:.4f}`")
                st.markdown("---")
