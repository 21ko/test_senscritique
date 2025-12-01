# Comment Similarity Finder 

Application Streamlit permettant :
- d’importer un fichier CSV ou d’utiliser deux fichiers bruts prédéfinis,
- de nettoyer les commentaires via un module dédié,
- de générer des embeddings SBERT,
- de comparer un commentaire utilisateur aux plus similaires,
- et d’afficher les Top-k résultats.



## Fonctionnalités

### 1. Upload ou sélection de fichiers
- Importation d’un CSV personnel
- Ou choix d’un des fichiers bruts :
  - `fightclub_critiques.csv`
  - `interstellar_critique.csv`

### 2. Nettoyage automatique  
Utilisation d’une classe `Cleaner` pour :
- suppression de la ponctuation
- normalisation du texte
- tokenization légère
- suppression des mots inutiles

### 3. Embeddings SBERT
Utilisation de **Sentence-BERT** via `sentence-transformers` pour convertir les commentaires en vecteurs (768 dimensions).

### 4. Similarité cosinus
Pour trouver les **commentaires les plus proches sémantiquement**.

### 5. Interface Streamlit
Expérience simple :
- choix du mode (taper un texte ou sélection d’un commentaire)
- bouton “Trouver les commentaires similaires”
- affichage des scores et commentaires

---

## Technologies utilisées

| Composant | Technologie |
|----------|-------------|
| Interface | Streamlit |
| NLP | Sentence-BERT (SBERT) |
| Similarité | Cosine Similarity |
| Nettoyage | Module Python personnalisé |
| Données | CSV uploadés ou fichiers bruts |

---


---





