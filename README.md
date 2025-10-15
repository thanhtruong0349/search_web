# 👨‍🔬📚 Academic Paper Search Engine 

## Overview

With the increasing number of scientific articles on open data repositories such as arXiv, IEEE, ACM, and Springer, finding relevant, accurate, and fast documents has become a big challenge for researchers. Existing tools such as Google Scholar or Semantic Scholar are powerful but limited in customization and lack transparency in ranking algorithms.
This project aims to develop a search platform that can: preprocess text, inverted index, implement and compare Information Retrieval (IR) models such as Boolean Retrieval, Vector Space Model (VSM), Okapi BM25, and BERT.
The system is built with a Web interface (Flask), allowing users to enter queries, select search models, and filter results by author, topic, or publication time. The results are evaluated using standard metrics such as Precision, Recall, and F1-score to compare the performance between models.

## 🎥 Preview 
<!-- ![Search Engine Preview](/app%20screenshots/SearchEngineUsage.gif) -->
<video src="/app%20screenshots/demo.mp4" controls autoplay loop muted width="700">
  Your browser does not support the video tag.
</video>
## 🛠️ Getting Started 

### Prerequisites:
- Python 3.9 or higher
- Git

1. **Clone the Repository:**
     ```bash
     git clone https://github.com/thanhtruong0349/search_web
     cd sources
     ```
2. **Install Dependencies:**
     ```bash
     pip install -r requirements.txt
     ```
3. **Download nltk necessary resources:**
     ```bash
     python nltk_resources.py
     ```
4. **Create a .env file in the root directory of the project and configure it similar to the example.env:**
     - **FLASK_HOST:** Set the **host address** for your Flask application. If running locally, you can set this to localhost. If deploying to a server, use the server's IP address or domain name.
     - **FLASK_PORT:** Define the **port number** where your Flask application will run. The default Flask port is 5000.
     - **FLASK_DEBUG:** Enable or disable Flask's debugging mode. When enabled, Flask will reload itself on code changes and provide more detailed error messages. Set to **1/True** to enable debugging, or **0/False** to disable it.
     - **DATASET_PATH:** Specify the **local file path** to the dataset that your Flask application will use.
     
5. **Run the Application:**
     ```bash
     python app.py
     ```
     The web interface should now be accessible at `http://localhost:5000` in your web browser.
6. **Perform a Search:**
   -  Perform a search using the user-friendly interface.
   -  Choose a retrieval algorithm from the dropdown list.
   -  Use filtering options to refine your search.

## 🔮💡 Future Ideas 

In the next phase, the team plans to expand the system in the following directions:
- Developing keyword pair and natural semantic search: Allowing users to query by keyword pair, natural language instead of just individual keywords.
- Fine-tuning the academic BERT model: Using variants like SciBERT or SPECTRE to improve search efficiency on scientific research data.
- Integrating RAG (Retrieval-Augmented Generation): Combining the retrieval model with LLM to generate feedback, summarizing information from many articles.

## 🙏 Acknowledgements 

- Thank you to arXiv for use of its open access interoperability.


