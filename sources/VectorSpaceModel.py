from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity

class VectorSpaceModel:
    def __init__(self, documents):
        self.documents = documents                      # dict[doc_id -> text]
        self.doc_ids = list(self.documents.keys())      # lưu danh sách doc_id
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

        # Fit TF-IDF ngay khi khởi tạo
        if self.documents:
            self.build_tfidf_matrix()

    def add_document(self, doc_id, text):
        """Thêm document mới và refit TF-IDF"""
        self.documents[doc_id] = text
        self.doc_ids.append(doc_id)
        self.build_tfidf_matrix()

    def build_tfidf_matrix(self):
        """Xây dựng TF-IDF matrix cho toàn bộ documents"""
        all_texts = list(self.documents.values())
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)

    def calculate_similarity(self, query):
        """Tính cosine similarity giữa query và tất cả document"""
        if self.tfidf_matrix is None:
            self.build_tfidf_matrix()
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        return similarities
    
    def retrieve(self, query, top_k=10):
        similarities = self.calculate_similarity(query)
        results = [(self.doc_ids[i], float(score)) for i, score in enumerate(similarities)]
        results = sorted(results, key=lambda x: x[1], reverse=True)
        print("DEBUG VSM all:", results[:10])
        return results[:top_k]   # luôn (doc_id, score)
