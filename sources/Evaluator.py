# # from file_operations import retrieve_data
# # from text_processing import preprocess_paper
# # from sklearn.metrics import precision_score, recall_score, f1_score
# # from SearchEngine import SearchEngine
# # import matplotlib.pyplot as plt
# # class Evaluator:
# #     def __init__(self, search_engine, ground_truth):
# #         self.search_engine = search_engine
# #         self.ground_truth = ground_truth

# #     def evaluate(self, queries):
# #         results = {
# #             'boolean': [],
# #             'vsm': [],
# #             'okapi_bm25': [],
# #         }
# #         for query in queries:
# #             boolean_papers, vsm_papers, okapi_papers = self.search_engine.search(query)

# #             # Evaluate Boolean retrieval
# #             boolean_metrics = self.calculate_metrics(boolean_papers)
# #             results['boolean'].append(boolean_metrics)

# #             # Evaluate VSM retrieval
# #             vsm_metrics = self.calculate_metrics(vsm_papers)
# #             results['vsm'].append(vsm_metrics)

# #             # Evaluate Okapi BM25 retrieval
# #             okapi_metrics = self.calculate_metrics(okapi_papers)
# #             results['okapi_bm25'].append(okapi_metrics)
# #         return results

# #     def calculate_metrics(self, retrieved_docs):
# #         y_true = [1 if doc_id in self.ground_truth else 0 for doc_id in retrieved_docs]
# #         y_pred = [1] * len(retrieved_docs)

# #         precision = precision_score(y_true, y_pred, zero_division=0)
# #         recall = recall_score(y_true, y_pred, zero_division=0)
# #         f1 = f1_score(y_true, y_pred, zero_division=0)

# #         return {
# #             'precision': precision,
# #             'recall': recall,
# #             'f1': f1,
# #         }
# #     def plot_metrics(self,algorithm, results, queries):
# #         for metric_name in ['precision', 'recall', 'f1']:
# #             metric_values = [metrics[metric_name] for metrics in results[algorithm]]
# #             plt.figure(figsize=(10, 5))
# #             plt.bar(queries, metric_values, color='blue', alpha=0.7)
# #             plt.title(f"{algorithm} {metric_name.capitalize()} Scores")
# #             plt.xlabel('Queries')
# #             plt.ylabel(metric_name.capitalize())
# #             plt.tight_layout()
# #             plt.show()


# # papers_collection = retrieve_data('../datasets/arXiv_papers_less.json')
# # preprocessed_metadata = {}
# # for paper in papers_collection:
# #     document_id = paper['arXiv ID']
# #     preprocessed_metadata[document_id] = preprocess_paper(paper)

# # search_engine = SearchEngine(preprocessed_metadata)
# # search_engine.build_inverted_index()
# # queries = [
# #     "mitigating over-smoothing regularized nonlocal functionals",
# #     "scalable meta-learning gaussian processes",
# #     "infrared super-resolution gan",
# #     "how tune autofocals comparative study",
# #     "efficiency spectrum algorithmic survey",
# #     "generalized label-efficient scene parsing hierarchical",
# #     "resource-constrained knowledge diffusion processes inspired",
# #     "maxmem colocation performance big applications",
# #     "rethinking domain gap near-infrared face",
# #     "weighted riesz particles",
# #     "experiment gender racial ethnic bias",
# #     "bcn batch channel normalization classification",
# #     "generative visualising abstract social processes",
# #     "hiding plain sight security defences",
# #     "clustering contour coreset variational quantum",
# #     "inherent limitations llms regarding spatial",
# #     "evaluating creativity literary perspective",
# #     "reversible entanglement beyond quantum operations",
# #     "which linguistic cues make people",
# #     "stochastic-constrained stochastic optimization markovian",
# #     "training synthetic beats real multimodal",
# #     "advancements trends ultra-high-resolution overview",
# #     "cascaded channel decoupling solution ris",
# #     "medication abortion digital health united",
# #     "dns slam dense semantic-informed",
# #     "optimal attack defense reinforcement",
# #     "concept erasure kernelized rate-distortion maximization",
# #     "hetrinet heterogeneous triplet attention drug-target-disease",
# #     "pedaling fast slow race optimized",
# #     "deepen2023 energy edge artificial intelligence",
# #     "unsupervised representation evaluating transferring visual",
# #     "sparsedc depth completion sparse non-uniform",
# #     "graphdreamer compositional scene synthesis",
# #     "sound terminology describing production perception",
# #     "star colouring locally constrained homomorphisms",
# #     "compact implicit representation efficient storage",
# #     "pdb-struct comprehensive benchmark structure-based protein",
# #     "enhancing cross-domain click-through rate prediction",
# #     "binary perceptrons capacity fully lifted",
# #     "probabilistic copyright protection fail text-to-image",
# #     "leap llm-generation egocentric action programs",
# #     "who leading analysis industry research",
# #     "surreyai 2023 submission quality estimation",
# #     "summarization-based augmentation document classification",
# #     "bayesian causal discovery unknown general",
# #     "slotted aloha optical wireless communications",
# #     "semantics attack-defense trees dynamic countermeasures",
# #     "auto-encoding gps reveal individual collective",
# #     "fsgs real-time few-shot view synthesis",
# #     "consensus group decision making uncertainty"
# # ]
# # total_ground_truth = []
# # for query in queries:
# #     _,_,ground_truth = search_engine.search(query)
# #     total_ground_truth += ground_truth
# # evaluator = Evaluator(search_engine, total_ground_truth)
# # results = evaluator.evaluate(queries)
# # for algorithm, metrics_list in results.items():
# #     evaluator.plot_metrics(algorithm, results, queries)
    
# # for algorithm, metrics_list in results.items():
# #     print(f"\nMetrics for {algorithm} retrieval:")
# #     for i, metrics in enumerate(metrics_list, start=1):
# #         print(f"{queries[i-1]}: Precision={metrics['precision']}, Recall={metrics['recall']}, F1={metrics['f1']}")
# # -*- coding: utf-8 -*-
# """
# Đánh giá Boolean, VSM, BM25 bằng Precision / Recall / F1 theo từng truy vấn
# và macro-average, dựa trên queries.csv + qrels.txt.

# Yêu cầu:
# - SearchEngine của bạn:
#   - build_preprocessed_documents(papers)
#   - build_inverted_index()
#   - build_models()  # đã cài VSMRanker + OkapiBM25 như các bước trước
#   - search(query) -> (boolean_docids, vsm_pairs, bm25_pairs)
#     * boolean_docids: list[str]
#     * vsm_pairs / bm25_pairs: list[(docid:str, score:float)]
# """

# import csv
# import argparse
# from pathlib import Path
# from typing import Dict, List, Set, Tuple

# from file_operations import retrieve_data
# from SearchEngine import SearchEngine


# # --------------------------
# # Utils: load queries & qrels
# # --------------------------

# def load_queries_csv(path: str) -> Dict[str, str]:
#     """Đọc queries.csv (qid,query) -> dict[qid(str) -> query(str)]"""
#     out = {}
#     with open(path, newline="", encoding="utf-8") as f:
#         for row in csv.DictReader(f):
#             qid = str(row["qid"]).strip()
#             q = (row["query"] or "").strip()
#             if qid and q:
#                 out[qid] = q
#     return out


# def load_qrels(path: str) -> Dict[str, Set[str]]:
#     """
#     Đọc qrels.txt (TREC format): <qid> 0 <docid> <rel>
#     -> dict[qid -> set(docid)] chỉ giữ các docid có rel > 0
#     """
#     out: Dict[str, Set[str]] = {}
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split()
#             if len(parts) != 4:
#                 continue
#             qid, _, docid, rel = parts
#             try:
#                 rel = int(rel)
#             except Exception:
#                 rel = 0
#             if rel > 0:
#                 out.setdefault(qid, set()).add(docid)
#     return out


# # --------------------------
# # Metrics (per-query)
# # --------------------------

# def prf_scores(retrieved: List[str], relevant: Set[str]) -> Tuple[float, float, float]:
#     """
#     Precision / Recall / F1 cho một truy vấn.
#     - retrieved: list docid (top-k) trả về
#     - relevant:  set docid ground-truth của truy vấn
#     """
#     if not retrieved:
#         return 0.0, 0.0, 0.0
#     R = set(retrieved)
#     GT = set(relevant or [])
#     tp = len(R & GT)
#     fp = len(R - GT)
#     fn = len(GT - R)

#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
#     return precision, recall, f1


# # --------------------------
# # Evaluation pipeline
# # --------------------------

# def evaluate(models_results: Dict[str, Dict[str, List[str]]],
#              qrels: Dict[str, Set[str]]) -> Dict[str, Dict[str, float]]:
#     """
#     models_results: { model_name -> { qid -> [docid,...] } }
#     qrels:          { qid -> set(docid) }

#     Trả về: { model_name -> { 'P': macroP, 'R': macroR, 'F1': macroF1 } }
#     """
#     macro = {}
#     for model, perq in models_results.items():
#         P_list, R_list, F1_list = [], [], []
#         for qid, retrieved in perq.items():
#             p, r, f = prf_scores(retrieved, qrels.get(qid, set()))
#             P_list.append(p); R_list.append(r); F1_list.append(f)
#         # macro = trung bình các query
#         macro[model] = {
#             "P": sum(P_list) / len(P_list) if P_list else 0.0,
#             "R": sum(R_list) / len(R_list) if R_list else 0.0,
#             "F1": sum(F1_list) / len(F1_list) if F1_list else 0.0,
#         }
#     return macro


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", default="../datasets/arXiv_papers_less.json",
#                     help="Đường dẫn corpus arXiv JSON")
#     ap.add_argument("--queries", default="queries.csv",
#                     help="CSV: qid,query")
#     ap.add_argument("--qrels", default="qrels.txt",
#                     help="TREC qrels: <qid> 0 <docid> <rel>")
#     ap.add_argument("--topk", type=int, default=100,
#                     help="Lấy top-k để tính P/R/F1")
#     ap.add_argument("--out_csv", default="prf_results.csv",
#                     help="(Tùy chọn) Xuất per-query P/R/F1 ra CSV")
#     args = ap.parse_args()

#     # 1) Load data
#     papers = retrieve_data(args.dataset)

#     # 2) Build engine + models
#     se = SearchEngine()
#     se.build_preprocessed_documents(papers)
#     se.build_inverted_index()
#     # RẤT QUAN TRỌNG: đã chỉnh SearchEngine để có build_models()
#     se.build_models()

#     # 3) Load queries & qrels
#     queries = load_queries_csv(args.queries)        # dict[qid->query]
#     qrels   = load_qrels(args.qrels)                # dict[qid->set(docid)]

#     # 4) Chạy search & gom kết quả về dạng list[docid]
#     results: Dict[str, Dict[str, List[str]]] = {
#         "Boolean": {},
#         "VSM": {},
#         "BM25": {},
#     }
#     for qid, query in queries.items():
#         boolean_ids, vsm_pairs, bm25_pairs = se.search(query)

#         results["Boolean"][qid] = list(boolean_ids or [])

#         vsm_ids  = [doc for doc, _ in (vsm_pairs or [])][:args.topk]
#         bm25_ids = [doc for doc, _ in (bm25_pairs or [])][:args.topk]
#         results["VSM"][qid]  = vsm_ids
#         results["BM25"][qid] = bm25_ids

#     # 5) Tính P/R/F1 macro
#     macro = evaluate(results, qrels)

#     # 6) In kết quả
#     print("\n=== Macro Precision/Recall/F1 (top-{}) ===".format(args.topk))
#     for m in ["Boolean", "VSM", "BM25"]:
#         s = macro.get(m, {})
#         print(f"{m:8s}  P={s.get('P',0):.4f}  R={s.get('R',0):.4f}  F1={s.get('F1',0):.4f}")

#     # 7) (Tùy chọn) Xuất per-query P/R/F1
#     if args.out_csv:
#         out_path = Path(args.out_csv)
#         with out_path.open("w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(["qid", "model", "precision", "recall", "f1"])
#             for model, perq in results.items():
#                 for qid, retrieved in perq.items():
#                     p, r, f = prf_scores(retrieved, qrels.get(qid, set()))
#                     w.writerow([qid, model, f"{p:.6f}", f"{r:.6f}", f"{f:.6f}"])
#         print(f"\nĐã xuất per-query P/R/F1 -> {out_path.resolve()}")


# if __name__ == "__main__":
#     main()

# -*- coding: utf-8 -*-
"""
eval.py — Đánh giá Boolean, VSM, BM25 bằng Precision / Recall / F1.
Phù hợp với SearchEngine bạn cung cấp:
    - build_preprocessed_documents(papers_collection)
    - build_inverted_index()
    - search(query) -> (boolean_results, vsm_results, okapiBM25_results)

Yêu cầu dữ liệu:
- queries.csv   : CSV có cột 'qid,query'
- qrels.txt     : TREC qrels: "<qid> 0 <docid> <rel>" (rel > 0 coi là liên quan)

Cách chạy (ví dụ):
python eval.py \
  --dataset ../datasets/arXiv_papers_less.json \
  --queries queries.csv \
  --qrels qrels.txt \
  --topk 100 \
  --per_query_csv per_query_prf.csv
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Any

from file_operations import retrieve_data
from SearchEngine import SearchEngine


# =========================
# I. Helpers: load dữ liệu
# =========================

def load_queries_csv(path: str) -> Dict[str, str]:
    """Đọc queries.csv (qid,query) -> dict[qid -> query]"""
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("qid", "")).strip()
            q   = str(row.get("query", "")).strip()
            if qid and q:
                out[qid] = q
    return out


def load_qrels(path: str) -> Dict[str, Set[str]]:
    """
    Đọc qrels.txt theo TREC format:
        <qid> 0 <docid> <rel>
    Trả về: dict[qid -> set(docid)] chỉ giữ docid có rel > 0
    """
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            try:
                rel = int(rel)
            except Exception:
                rel = 0
            if rel > 0:
                out.setdefault(qid, set()).add(docid)
    return out


# ==========================================
# II. Chuẩn hoá kết quả -> danh sách docid
# ==========================================

def _ids_from_iterable(x: Iterable[Any]) -> List[str]:
    """
    Chấp nhận nhiều kiểu kết quả:
      - list[str]                               -> giữ nguyên
      - list[tuple(docid, score)]               -> lấy docid
      - dict[docid -> score]                    -> lấy key
    """
    ids: List[str] = []
    for it in x:
        if isinstance(it, tuple) and len(it) >= 1:
            ids.append(str(it[0]))
        else:
            ids.append(str(it))
    return ids


def normalize_result_to_ids(result: Any, topk: int) -> List[str]:
    """
    Chuẩn hóa output của một mô hình về list[str] docid và cắt topk.
    Hỗ trợ:
      - list[str]
      - list[(docid, score)]
      - dict[docid -> score] (sẽ sắp xếp theo score giảm dần)
    """
    if result is None:
        return []
    # dict -> sort by score desc
    if isinstance(result, dict):
        items = sorted(result.items(), key=lambda kv: kv[1], reverse=True)
        return [str(doc) for doc, _ in items[:topk]]

    # list -> có thể là list[str] hoặc list[(docid, score)]
    if isinstance(result, list):
        ids = _ids_from_iterable(result)
        return ids[:topk]

    # fallback: cố gắng ép kiểu đơn giản
    try:
        return list(map(str, list(result)))[:topk]
    except Exception:
        return []


# ============================
# III. Tính Precision/Recall/F1
# ============================

def prf_scores(retrieved: List[str], relevant: Set[str]) -> Tuple[float, float, float]:
    """
    Precision / Recall / F1 cho một truy vấn.
    - retrieved: list docid (top-k) trả về
    - relevant:  set docid ground-truth của truy vấn
    """
    if not retrieved:
        return 0.0, 0.0, 0.0
    R = set(retrieved)
    GT = set(relevant or [])
    tp = len(R & GT)
    fp = len(R - GT)
    fn = len(GT - R)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_models(per_model_results: Dict[str, Dict[str, List[str]]],
                    qrels: Dict[str, Set[str]]) -> Dict[str, Dict[str, float]]:
    """
    per_model_results: { model_name -> { qid -> [docid,...] } }
    qrels           : { qid -> set(docid) }

    Trả về: { model_name -> { 'P': macroP, 'R': macroR, 'F1': macroF1 } }
    """
    macro: Dict[str, Dict[str, float]] = {}
    for model, perq in per_model_results.items():
        P_list: List[float] = []
        R_list: List[float] = []
        F1_list: List[float] = []
        for qid, retrieved in perq.items():
            p, r, f = prf_scores(retrieved, qrels.get(qid, set()))
            P_list.append(p); R_list.append(r); F1_list.append(f)
        macro[model] = {
            "P": sum(P_list) / len(P_list) if P_list else 0.0,
            "R": sum(R_list) / len(R_list) if R_list else 0.0,
            "F1": sum(F1_list) / len(F1_list) if F1_list else 0.0,
        }
    return macro


# ===========
# IV. main()
# ===========

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", default="../datasets/arXiv_papers_less.json",
#                     help="Đường dẫn corpus arXiv JSON (file_operations.retrieve_data)")
#     ap.add_argument("--queries", default="queries.csv",
#                     help="CSV có cột qid,query")
#     ap.add_argument("--qrels", default="qrels.txt",
#                     help="TREC qrels: <qid> 0 <docid> <rel>")
#     ap.add_argument("--topk", type=int, default=100,
#                     help="Cắt top-k cho việc tính P/R/F1")
#     ap.add_argument("--per_query_csv", default="",
#                     help="(Tùy chọn) Ghi P/R/F1 theo từng truy vấn ra CSV")
#     args = ap.parse_args()

#     # 1) Nạp dữ liệu & dựng chỉ mục
#     papers_collection = retrieve_data(args.dataset)
#     se = SearchEngine()
#     se.build_preprocessed_documents(papers_collection)
#     se.build_inverted_index()

#     # 2) Tải queries và qrels
#     queries = load_queries_csv(args.queries)    # dict[qid->query]
#     qrels   = load_qrels(args.qrels)            # dict[qid->set(docid)]

#     # 3) Chạy tìm kiếm với tất cả mô hình
#     per_model_results: Dict[str, Dict[str, List[str]]] = {
#         "Boolean": {},
#         "VSM": {},
#         "BM25": {},
#     }
#     for qid, query in queries.items():
#         boolean_res, vsm_res, bm25_res = se.search(query)
#         per_model_results["Boolean"][qid] = normalize_result_to_ids(boolean_res, args.topk)
#         per_model_results["VSM"][qid]     = normalize_result_to_ids(vsm_res,     args.topk)
#         per_model_results["BM25"][qid]    = normalize_result_to_ids(bm25_res,    args.topk)

#     # 4) Tính macro P/R/F1
#     macro = evaluate_models(per_model_results, qrels)

#     # 5) In kết quả tổng hợp
#     print("\n=== Macro Precision / Recall / F1 (top-{}) ===".format(args.topk))
#     for name in ["Boolean", "VSM", "BM25"]:
#         s = macro.get(name, {})
#         print(f"{name:8s}  P={s.get('P',0.0):.4f}  R={s.get('R',0.0):.4f}  F1={s.get('F1',0.0):.4f}")

#     # 6) (Tùy chọn) Xuất per-query P/R/F1
#     if args.per_query_csv:
#         out_path = Path(args.per_query_csv)
#         with out_path.open("w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(["qid", "model", "precision", "recall", "f1"])
#             for model, perq in per_model_results.items():
#                 for qid, retrieved in perq.items():
#                     p, r, f = prf_scores(retrieved, qrels.get(qid, set()))
#                     w.writerow([qid, model, f"{p:.6f}", f"{r:.6f}", f"{f:.6f}"])
#         print(f"\nĐã ghi per-query P/R/F1 -> {out_path.resolve()}")

# ===========================
# IV. main() chỉnh thêm BERT
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../datasets/arXiv_papers_less.json",
                    help="Đường dẫn corpus arXiv JSON (file_operations.retrieve_data)")
    ap.add_argument("--queries", default="queries.csv",
                    help="CSV có cột qid,query")
    ap.add_argument("--qrels", default="qrels.txt",
                    help="TREC qrels: <qid> 0 <docid> <rel>")
    ap.add_argument("--topk", type=int, default=100,
                    help="Cắt top-k cho việc tính P/R/F1")
    ap.add_argument("--per_query_csv", default="",
                    help="(Tùy chọn) Ghi P/R/F1 theo từng truy vấn ra CSV")
    args = ap.parse_args()

    # 1) Load data & build SearchEngine
    papers_collection = retrieve_data(args.dataset)
    se = SearchEngine()
    se.build_preprocessed_documents(papers_collection)
    se.build_inverted_index()
    
    se.init_query_processor()

    # ⚠️ SearchEngine nên gọi QueryProcessor bên bạn (có BERT)
    # Giả sử SearchEngine.search(query) -> (bool_res, vsm_res, bm25_res, bert_res)

    # 2) Load queries & qrels
    queries = load_queries_csv(args.queries)
    qrels   = load_qrels(args.qrels)

    # 3) Chạy tìm kiếm cho tất cả mô hình
    per_model_results: Dict[str, Dict[str, List[str]]] = {
        "Boolean": {},
        "VSM": {},
        "BM25": {},
        "BERT": {},   # ✅ thêm BERT
    }
    for qid, query in queries.items():
        boolean_res, vsm_res, bm25_res, bert_res = se.search(query)

        per_model_results["Boolean"][qid] = normalize_result_to_ids(boolean_res, args.topk)
        per_model_results["VSM"][qid]     = normalize_result_to_ids(vsm_res,     args.topk)
        per_model_results["BM25"][qid]    = normalize_result_to_ids(bm25_res,    args.topk)
        per_model_results["BERT"][qid]    = normalize_result_to_ids(bert_res,    args.topk)  # ✅

    # 4) Evaluate macro P/R/F1
    macro = evaluate_models(per_model_results, qrels)

    # 5) In kết quả
    print("\n=== Macro Precision / Recall / F1 (top-{}) ===".format(args.topk))
    for name in ["Boolean", "VSM", "BM25", "BERT"]:  # ✅ thêm BERT
        s = macro.get(name, {})
        print(f"{name:8s}  P={s.get('P',0.0):.4f}  R={s.get('R',0.0):.4f}  F1={s.get('F1',0.0):.4f}")

    # 6) Xuất per-query CSV
    if args.per_query_csv:
        out_path = Path(args.per_query_csv)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["qid", "model", "precision", "recall", "f1"])
            for model, perq in per_model_results.items():
                for qid, retrieved in perq.items():
                    p, r, f = prf_scores(retrieved, qrels.get(qid, set()))
                    w.writerow([qid, model, f"{p:.6f}", f"{r:.6f}", f"{f:.6f}"])
        print(f"\nĐã ghi per-query P/R/F1 -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
