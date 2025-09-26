from collections import defaultdict
import re

# ---- Chuẩn hoá dùng chung cho cả index & truy vấn ----
def normalize(text: str) -> str:
    # hạ chữ, bỏ ký tự không phải chữ/số, tách theo khoảng trắng
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return text

def tokenize_text(text: str):
    return [t for t in normalize(text).split() if t]

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)

    def add_document(self, doc_id, preprocessed_text):
        tokens = tokenize_text(preprocessed_text)
        term_freq = defaultdict(int)
        for i, token in enumerate(tokens):
            term_freq[token] += 1
            # lưu vị trí
            if doc_id not in self.index[token]:
                self.index[token][doc_id] = {'frequency': 0, 'positions': []}
            self.index[token][doc_id]['positions'].append(i)

        # cập nhật tần suất
        for term, freq in term_freq.items():
            self.index[term][doc_id]['frequency'] += freq

    def get_term_info(self, term):
        term = normalize(term).strip()       # <<< CHUẨN HOÁ Ở ĐÂY
        return self.index.get(term, {})

    def print_terms(self, terms):
        # bỏ trùng, vẫn giữ thứ tự truyền vào
        unique_terms = list(dict.fromkeys(terms))
        for term in unique_terms:
            term_info = self.get_term_info(term)
            print(f"Inverted Index for term '{term}':")
            if not term_info:
                print("  (no documents)")
            else:
                sorted_docs = sorted(
                    term_info.items(),
                    key=lambda x: x[1]['frequency'],
                    reverse=True
                )
                for doc_id, doc_info in sorted_docs:
                    print(f"  Document ID: {doc_id}, "
                          f"Frequency: {doc_info['frequency']}, "
                          f"Positions: {doc_info['positions']}")
            print("="*50)

# ------------------- DEMO -------------------
if __name__ == "__main__":
    index = InvertedIndex()
    docs = {
        1: "Artificial Intelligence and Machine Learning",
        2: "Machine learning is a subfield of artificial intelligence",
        3: "Deep Learning advances Machine Intelligence"
    }

    for doc_id, text in docs.items():
        tokens = tokenize_text(text)
        print(f"Doc {doc_id} raw: {text}")
        print(f"Doc {doc_id} preprocessed tokens: {tokens}\n")
        index.add_document(doc_id, text)

    # Truy vấn
    index.print_terms(["Intelligence", "Artificial", "Learning", "Machine"])
