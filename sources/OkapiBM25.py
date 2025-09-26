# import math
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from text_processing import tokenize_text
# class OkapiBM25:
#     def __init__(self, documents, k1=1.5, b=0.75):
#         self.k1 = k1
#         self.b = b
#         self.avg_doc_length = 0
#         self.doc_lengths = {}
#         self.idf = {}
#         self.vectorizer = TfidfVectorizer()
#         self.documents = documents
#         # Initialize the Okapi BM25 model with the provided documents
#         for doc_id, text in self.documents.items():
#             self.add_document(doc_id, text)

#     def add_document(self, doc_id, text):
#         # Tokenize the text
#         tokens = tokenize_text(text)
#         # Calculate document length
#         doc_length = len(tokens)
#         self.doc_lengths[doc_id] = doc_length
#         # Update average document length
#         self.avg_doc_length = (self.avg_doc_length * (len(self.doc_lengths) - 1) + doc_length) / len(self.doc_lengths)
#         # Update document frequency (DF) and inverse document frequency (IDF) for each term
#         term_counts = Counter(tokens)
#         for term, count in term_counts.items():
#             if term not in self.idf:
#                 self.idf[term] = 1
#             self.idf[term] += 1

#     def calculate_bm25_score(self, query, doc_id):
#         # Tokenize the query
#         query_tokens = tokenize_text(query)
#         # Get document length
#         doc_length = self.doc_lengths.get(doc_id, 0)
#         # Calculate BM25 score
#         score = 0
#         for term in query_tokens:
#             tf = Counter(self.vectorizer.build_analyzer()(self.documents[doc_id]))[term]
#             idf = math.log((len(self.doc_lengths) - self.idf.get(term, 0) + 0.5) / (self.idf.get(term, 0) + 0.5) + 1)
#             numerator = tf * (self.k1 + 1)
#             denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
#             score += idf * numerator / denominator
#         return score
#     def retrieve(self, query,score_threshold=1.0):
#         scores = [(doc_id, self.calculate_bm25_score(query, doc_id)) for doc_id in self.documents]
#         sorted_results = sorted(scores, key=lambda x: x[1], reverse=True) #Sort the results by similarity scores
#         matching_documents = [doc_id for doc_id, score in sorted_results if score >= score_threshold]
#         return matching_documents


import math
from collections import Counter, defaultdict
from text_processing import tokenize_text

class OkapiBM25:
    """
    Okapi BM25 (đúng công thức), giữ interface cũ:
      - __init__(documents, k1=1.5, b=0.75)
      - add_document(doc_id, text)
      - calculate_bm25_score(query, doc_id)
      - retrieve(query, score_threshold=1.0) -> list[doc_id] (sorted desc)
    """
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.documents = {}            # doc_id -> raw/preprocessed text
        self.term_freqs = {}           # doc_id -> Counter(term -> tf)
        self.doc_lengths = {}          # doc_id -> int
        self.df = Counter()            # term -> document frequency
        self.idf = {}                  # term -> idf
        self.total_len = 0             # tổng số token toàn bộ corpora
        self.N = 0                     # số tài liệu
        self.avg_doc_length = 0.0

        # nạp tài liệu ban đầu
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def _recompute_idf(self):
        # Biến thể IDF hay dùng: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {}
        if self.N == 0:
            return
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def add_document(self, doc_id, text):
        """Có thể gọi sau khi init để bổ sung tài liệu mới."""
        # nếu doc_id đã tồn tại, bỏ cũ ra khỏi thống kê trước (đảm bảo idempotent)
        if doc_id in self.documents:
            old_tf = self.term_freqs.get(doc_id, Counter())
            # trừ DF cho các term cũ
            for term in old_tf.keys():
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]
            # trừ chiều dài & N sẽ cập nhật lại bên dưới
            self.total_len -= self.doc_lengths.get(doc_id, 0)
            self.N -= 1

        self.documents[doc_id] = text

        tokens = tokenize_text(text)              # quan trọng: cùng tokenizer với truy vấn
        tf = Counter(tokens)
        self.term_freqs[doc_id] = tf

        dl = len(tokens)
        self.doc_lengths[doc_id] = dl
        self.total_len += dl

        # tăng DF mỗi term một lần cho tài liệu này
        for term in tf.keys():
            self.df[term] += 1

        self.N += 1
        self.avg_doc_length = (self.total_len / self.N) if self.N else 0.0
        self._recompute_idf()

    def calculate_bm25_score(self, query, doc_id):
        if doc_id not in self.documents:
            return 0.0

        q_tokens = tokenize_text(query)
        if not q_tokens:
            return 0.0

        tf_doc = self.term_freqs.get(doc_id, {})
        dl = self.doc_lengths.get(doc_id, 0)
        if dl == 0 or self.avg_doc_length == 0:
            return 0.0

        score = 0.0
        K = self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_length))

        # Có thể tính qf nếu muốn (Counter), ở đây coi mỗi term 1 lần cho đơn giản
        for term in q_tokens:
            tf = tf_doc.get(term, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(term, 0.0)  # term chưa thấy trong tập -> 0
            score += idf * ((tf * (self.k1 + 1)) / (tf + K))

        return score

    def retrieve(self, query, score_threshold=1.0):
        """
        GIỮ NGUYÊN HÀNH VI CŨ:
        - Trả về: list[doc_id] đã sắp xếp theo điểm giảm dần
        - Lọc theo ngưỡng score_threshold
        """
        scored = [(doc_id, self.calculate_bm25_score(query, doc_id))
                  for doc_id in self.documents.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        matching_documents = [doc_id for doc_id, s in scored if s >= score_threshold]
        return matching_documents

    # (tuỳ chọn) nếu web bạn sau này muốn hiện điểm:
    def retrieve_with_scores(self, query, top_k=None):
        scored = [(doc_id, self.calculate_bm25_score(query, doc_id))
                  for doc_id in self.documents.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k] if top_k else scored
