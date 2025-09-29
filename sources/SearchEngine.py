from QueryProcessor import QueryProcessor
from text_processing import preprocess_paper
from InvertedIndex import InvertedIndex
from RankingAlgorithms import RankingAlgorithm

class SearchEngine:
    def __init__(self, preprocessed_documents=None, inverted_index=None):
        self._current_query = None
        self.inverted_index = inverted_index
        self.preprocessed_documents = preprocessed_documents
        self.ranking = RankingAlgorithm()
        self.query_processor = None   # chỉ khởi tạo 1 lần sau khi build xong

    def build_preprocessed_documents(self, papers_collection):
        """Tiền xử lý toàn bộ papers"""
        preprocessed_documents = {}
        for paper in papers_collection:
            document_id = paper['arXiv ID']
            preprocessed_documents[document_id] = preprocess_paper(paper)
        self.preprocessed_documents = preprocessed_documents

    def build_inverted_index(self):
        """Tạo inverted index"""
        inverted_index = InvertedIndex()
        for document_id, preprocessed_text in self.preprocessed_documents.items():
            self.preprocessed_documents[document_id] = preprocessed_text
            inverted_index.add_document(document_id, preprocessed_text)
        self.inverted_index = inverted_index

    def init_query_processor(self, device="cpu"):
        """Khởi tạo QueryProcessor 1 lần duy nhất"""
        if self.query_processor is None:
            self.query_processor = QueryProcessor(
                self.inverted_index,
                self.preprocessed_documents,
                device=device
            )

    def search(self, query, top_k=10):
        """Chạy tìm kiếm trên 4 mô hình"""
        self._current_query = query
        if self.query_processor is None:
            raise ValueError("QueryProcessor chưa được khởi tạo. Hãy gọi init_query_processor().")
        boolean_results, vsm_results, bm25_results, bert_results = \
            self.query_processor.process_query(self._current_query, top_k=top_k)
            
        print("=== DEBUG Boolean all ===")
        print(boolean_results)

        print("=== DEBUG BERT all ===")
        print(bert_results)
        
        return boolean_results, vsm_results, bm25_results, bert_results

    def rank_results(self, results, query):
        """Xếp hạng lại kết quả bằng TF-IDF ranking"""
        self._current_query = query
        ranked_results = {}
        for result_id in results:
            ranked_results[result_id] = self.preprocessed_documents[result_id]
        ranked_results = self.ranking.tf_idf_ranking(ranked_results, self._current_query)
        return ranked_results

    def filter_results(self, results, filters, papers_collection):
        """Lọc kết quả theo filters"""
        filtered_papers = []
        for arxiv_id in results:
            paper = next((p for p in papers_collection if p['arXiv ID'] == arxiv_id), None)
            if paper:
                filter_match = all(
                    (value.lower() in map(str.lower, paper[key]))
                    if isinstance(paper[key], list) and key != 'Authors' else
                    any(value.lower() in author.lower() for author in paper[key].split(', '))
                    if ['Authors', 'Subjects', 'Subject_Tags', 'Submitted Date'] else
                    paper[key].lower() == value.lower()
                    for key, value in filters.items()
                )
                if filter_match:
                    filtered_papers.append(paper)
        return filtered_papers

    def get_current_query(self):
        return self._current_query

    def display_results(self, results):
        """In kết quả ra console (debug)"""
        for paper in results:
            print(f"arXiv ID: {paper['arXiv ID']}")
            print(f"Title: {paper['Title']}")
            print(f"Authors: {paper['Authors']}")
            print(f"Subject Tags: {paper['Subject_Tags']}")
            print(f"Subjects: {paper['Subjects']}")
            print(f"Submitted Date: {paper['Submitted Date']}")
            print(f"Abstract: {paper['Abstract']}")
            print(f"PDF Link: {paper['PDF Link']}")
            print(f"{'-'*200}")
