import math
from collections import Counter
from text_processing import tokenize_text

class OkapiBM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

        self.documents = {}            # doc_id -> text
        self.term_freqs = {}           # doc_id -> Counter(term -> tf)
        self.doc_lengths = {}          # doc_id -> int
        self.df = Counter()            # term -> document frequency
        self.idf = {}                  # term -> idf
        self.total_len = 0
        self.N = 0
        self.avg_doc_length = 0.0

        # Nạp tài liệu ban đầu
        for doc_id, text in documents.items():
            self.add_document(doc_id, text)

    def _recompute_idf(self):
        self.idf = {}
        if self.N == 0:
            return
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def add_document(self, doc_id, text):
        # Nếu doc_id đã tồn tại thì bỏ thống kê cũ
        if doc_id in self.documents:
            old_tf = self.term_freqs.get(doc_id, Counter())
            for term in old_tf.keys():
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]
            self.total_len -= self.doc_lengths.get(doc_id, 0)
            self.N -= 1

        self.documents[doc_id] = text
        tokens = tokenize_text(text)
        tf = Counter(tokens)
        self.term_freqs[doc_id] = tf

        dl = len(tokens)
        self.doc_lengths[doc_id] = dl
        self.total_len += dl

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

        for term in q_tokens:
            tf = tf_doc.get(term, 0)
            if tf <= 0:
                continue
            idf = self.idf.get(term, 0.0)
            score += idf * ((tf * (self.k1 + 1)) / (tf + K))

        return score
    
    def retrieve(self, query, top_k=10):
        scored = [(doc_id, self.calculate_bm25_score(query, doc_id))
                for doc_id in self.documents.keys()]
        scored.sort(key=lambda x: x[1], reverse=True)
        print("DEBUG BM25 all:", scored[:10])
        return scored[:top_k]   # luôn (doc_id, score)
