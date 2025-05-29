
# 📚 Multi-Lingual Assitance for Rural India

This project implements an intelligent FAQ retrieval system that returns the most relevant answer to a user query using semantic similarity techniques. The system employs **Word2Vec** and **GloVe** embeddings to understand the context and meaning of questions, even if they are phrased differently.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Preprocessing Techniques](#preprocessing-techniques)
- [Word Embeddings Used](#word-embeddings-used)
- [How It Works](#how-it-works)
- [User Interface](#user-interface)
- [Testing Video](#testing-video)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

Users often ask similar questions in different ways. A traditional keyword-based search might fail to retrieve the right answer. This project solves that by converting both FAQ entries and user queries into **semantic vectors** using **Word2Vec** and **GloVe**, enabling **context-aware** retrieval based on **cosine similarity**.

---

## 🌟 Features

- Converts questions into vector representations using Word2Vec and GloVe
- Measures similarity using cosine distance
- Displays the most relevant FAQ entry
- Built with an intuitive **Streamlit** interface
- Supports real-time querying

---

## 💻 Tech Stack

| Technology     | Description                        |
|----------------|------------------------------------|
| Python         | Core programming language          |
| Pandas         | Data manipulation and storage      |
| NLTK           | Natural language preprocessing     |
| Gensim         | Word2Vec modeling                  |
| NumPy          | Numerical vector operations        |
| Scikit-learn   | Cosine similarity calculation      |
| Streamlit      | Web interface for user interaction |

---

## ⚙️ Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/faq-retrieval-nlp.git
cd faq-retrieval-nlp
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Download GloVe Embeddings**
Download `glove.6B.100d.txt` from [GloVe Website](https://nlp.stanford.edu/projects/glove/) and place it in the `/data/` directory.

4. **Run the App**
```bash
streamlit run app.py
```

---

## 🔄 Preprocessing Techniques

All questions undergo:
- Lowercasing
- Tokenization
- Stopword Removal
- Lemmatization

```python
faq_df['processed_questions'] = faq_df['Question'].apply(
    lambda text: [lemmatizer.lemmatize(word) for word in word_tokenize(text.lower())
                  if word.isalpha() and word not in stop_words]
)
```

---

## 📈 Word Embeddings Used

### Word2Vec
- Trained on the FAQ questions locally.
- Captures contextual similarity using local window context.

```python
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
```

### GloVe
- Pre-trained model on global corpus (Wikipedia + Gigaword).
- Captures global co-occurrence.

```python
glove_model = {}
with open("data/glove.6B.100d.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        vector = np.array(parts[1:], dtype='float32')
        glove_model[word] = vector
```

---

## 🔍 How It Works

1. Preprocess user query
2. Convert it to Word2Vec and GloVe vectors
3. Calculate cosine similarity with stored FAQ vectors
4. Display the most similar FAQ answer

```python
tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(translated_query.lower())
          if word.isalpha() and word not in stop_words]

vec_w2v = get_word2vec_vector(tokens).reshape(1, -1)
vec_glove = get_glove_vector(tokens).reshape(1, -1)

sims_w2v = cosine_similarity(np.vstack(faq_df['word2vec_vectors'].values), vec_w2v).flatten()
sims_glove = cosine_similarity(np.vstack(faq_df['glove_vectors'].values), vec_glove).flatten()
faq_df['similarity'] = (sims_w2v + sims_glove) / 2

best_match = faq_df.loc[faq_df['similarity'].idxmax()]
```

---

## 🖼️ User Interface

Place your screenshots in a folder named `/screenshots/` and reference them like this:

```markdown
### 💡 Homepage
![Homepage](screenshots/homepage.png)

### 🔎 Query Result
![Query Result](screenshots/query_result.png)
```

---

## 🎥 Testing Video

Uploaded a demo video showing the system in action. Embed like this:

[![Watch the demo](https://youtu.be/qd2ziS6oeRA)](https://youtu.be/qd2ziS6oeRA)

---

## Get Your Google API Key
You need an API key from Google AI Studio to use the Gemini model.
#### How to Get Your API Key from Google AI Studio:
```text
1. Go to Google AI Studio: https://aistudio.google.com/
2. Sign in with your Google account.
3. Click on "API Access" (Top-right corner).
4. Click "Generate API Key".
5. Copy the generated API Key.
```

Replace the "YOUR_GEMINI_API_KEY" with your GEMINI API Key in the streamlit_app.py 

## 🤝 Contributing

Contributions are welcome! Open issues or pull requests for improvements.



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
