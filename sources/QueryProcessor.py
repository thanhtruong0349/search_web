
# from text_processing import preprocess_text, tokenize_text
# from VectorSpaceModel import VectorSpaceModel
# from OkapiBM25 import OkapiBM25
# from sentence_transformers import SentenceTransformer, util
# import torch

# class QueryProcessor:
#     def __init__(self, inverted_index, preprocessed_docs):
#         self.inverted_index = inverted_index
#         self.vector_space_model = VectorSpaceModel(preprocessed_docs)
#         self.okapi_bm25 = OkapiBM25(preprocessed_docs)

#         # --- BERT model ---
#         self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.doc_embeddings = {}
#         for doc_id, text in preprocessed_docs.items():
#             self.doc_embeddings[doc_id] = self.bert_model.encode(
#                 text, convert_to_tensor=True
#             )

#     def process_query(self, user_query):
#         query_tokens = tokenize_text(preprocess_text(user_query))

#         # VSM
#         vsm_results = self.vector_space_model.retrieve(preprocess_text(user_query), 0.05)

#         # BM25
#         bm25_results = self.okapi_bm25.retrieve(preprocess_text(user_query), 3.0)

#         # Boolean
#         boolean_results = self.boolean_retrieval(query_tokens)

#         # BERT
#         bert_results = self.bert_search(user_query, top_k=10)

#         return boolean_results, vsm_results, bm25_results, bert_results

#     def boolean_retrieval(self, processed_query_tokens):
#         current_documents = None  # initialize current docs with None
#         current_operator = 'AND'  # Default to AND

#         for token in processed_query_tokens:
#             if token.upper() in {'AND', 'OR', 'NOT'}:
#                 # Update the current operator
#                 current_operator = token.upper()
#             else:
#                 term_documents = set(self.inverted_index.index.get(token, {}).keys())
#                 # Update the set of current documents based on the current operator
#                 if current_documents is None:
#                     current_documents = term_documents
#                 elif current_operator == 'AND':
#                     current_documents = current_documents.intersection(term_documents)
#                 elif current_operator == 'OR':
#                     current_documents = current_documents.union(term_documents)
#                 elif current_operator == 'NOT':
#                     current_documents = current_documents.difference(term_documents)

#         return current_documents if current_documents is not None else set()

#     def bert_search(self, query, top_k=10):
#         query_embedding = self.bert_model.encode(query, convert_to_tensor=True)

#         doc_ids = list(self.doc_embeddings.keys())
#         # stack tất cả embedding thành 1 tensor
#         embeddings = torch.stack(list(self.doc_embeddings.values()))

#         # cosine similarity
#         scores = util.cos_sim(query_embedding, embeddings)[0]

#         # sắp xếp theo độ tương đồng
#         scored_docs = sorted(
#             zip(doc_ids, scores.tolist()),  # chuyển tensor -> float
#             key=lambda x: x[1],
#             reverse=True
#         )

#         return [doc_id for doc_id, score in scored_docs[:top_k]]

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

        # Tạo embeddings cho toàn bộ document (encode 1 lần duy nhất)
        self.doc_ids = list(preprocessed_docs.keys())
        doc_texts = list(preprocessed_docs.values())
        doc_embeddings = self.bert_model.encode(doc_texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)

        # Lưu FAISS index
        d = doc_embeddings.shape[1]  # kích thước vector
        self.index = faiss.IndexFlatIP(d)  # Inner Product ~ cosine nếu normalize
        self.index.add(doc_embeddings)

        self.doc_embeddings = doc_embeddings  # lưu numpy array để dùng khi cần

    def process_query(self, user_query):
        query_tokens = tokenize_text(preprocess_text(user_query))

        # VSM
        vsm_results = self.vector_space_model.retrieve(preprocess_text(user_query), 0.05)

        # BM25
        bm25_results = self.okapi_bm25.retrieve(preprocess_text(user_query), 3.0)

        # Boolean
        boolean_results = self.boolean_retrieval(query_tokens)

        # BERT (FAISS)
        bert_results = self.bert_search(user_query, top_k=10)

        return boolean_results, vsm_results, bm25_results, bert_results

    def boolean_retrieval(self, processed_query_tokens):
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

        return current_documents if current_documents is not None else set()

    def bert_search(self, query, top_k=10):
        query_emb = self.bert_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)

        # FAISS search
        D, I = self.index.search(query_emb, top_k)  # D = score, I = index
        results = [self.doc_ids[i] for i in I[0]]

        return results
