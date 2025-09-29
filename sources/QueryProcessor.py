from text_processing import preprocess_text, tokenize_text
from VectorSpaceModel import VectorSpaceModel
from OkapiBM25 import OkapiBM25
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class QueryProcessor:
    def __init__(self, inverted_index, preprocessed_docs, device="cpu"):
        self.inverted_index = inverted_index
        self.vector_space_model = VectorSpaceModel(preprocessed_docs)
        self.okapi_bm25 = OkapiBM25(preprocessed_docs)

        # --- BERT model ---
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        # Encode toàn bộ document (chỉ làm 1 lần khi init)
        self.doc_ids = list(preprocessed_docs.keys())
        doc_texts = list(preprocessed_docs.values())
        doc_embeddings = self.bert_model.encode(
            doc_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # FAISS index
        d = doc_embeddings.shape[1]  # kích thước vector
        self.index = faiss.IndexFlatIP(d)  # Inner Product ~ cosine nếu normalize
        self.index.add(doc_embeddings)

        self.doc_embeddings = doc_embeddings
        
    def process_query(self, user_query, top_k=10):
        query_tokens = tokenize_text(preprocess_text(user_query))
        processed_query = preprocess_text(user_query)

        # VSM -> chỉ lấy doc_id
        vsm_results = [doc_id for doc_id, _ in self.vector_space_model.retrieve(processed_query, top_k=top_k)]

        # BM25 -> chỉ lấy doc_id
        bm25_results = [doc_id for doc_id, _ in self.okapi_bm25.retrieve(processed_query, top_k=top_k)]

        # Boolean
        boolean_results = self.boolean_retrieval(query_tokens, top_k=top_k)

        # BERT
        bert_results = self.bert_search(user_query, top_k=top_k)

        return boolean_results, vsm_results, bm25_results, bert_results


    def boolean_retrieval(self, processed_query_tokens, top_k=10):
        current_documents = None
        current_operator = 'AND'

        for token in processed_query_tokens:
            if token.upper() in {'AND', 'OR', 'NOT'}:
                current_operator = token.upper()
            else:
                term_documents = set(self.inverted_index.index.get(token, {}).keys())
                if current_documents is None:
                    current_documents = term_documents
                elif current_operator == 'AND':
                    current_documents = current_documents.intersection(term_documents)
                elif current_operator == 'OR':
                    current_documents = current_documents.union(term_documents)
                elif current_operator == 'NOT':
                    current_documents = current_documents.difference(term_documents)

        current_documents = list(current_documents) if current_documents else []
        return current_documents[:top_k]

    def bert_search(self, query, top_k=10):
        query_emb = self.bert_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1)

        # FAISS search
        D, I = self.index.search(query_emb, top_k)  # D = score, I = index
        results = [self.doc_ids[i] for i in I[0]]

        return results

