# evaluate.py
import csv, argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from file_operations import retrieve_data
from SearchEngine import SearchEngine

# ====== IR metrics (pip install ir_measures) ======
from ir_measures import read_trec_qrels, read_trec_run, iter_calc, nDCG, MAP, MRR, P, Recall

def load_engine(dataset_path: str):
    papers = retrieve_data(dataset_path)
    se = SearchEngine()
    se.build_preprocessed_documents(papers)
    se.build_inverted_index()
    return se, papers

def get_ranked_results(se: "SearchEngine", papers: list, query: str, algorithm: str, topk: int):
    boolean, vsm, bm25 = se.search(query)
    if algorithm == "boolean":
        if boolean:
            ranked = se.rank_results(boolean, query)  # boolean cần xếp hạng
        else:
            ranked = []
    elif algorithm == "vsm":
        ranked = vsm or []
    elif algorithm == "okapiBM25":
        ranked = bm25 or []
    else:
        raise ValueError("algorithm must be one of: boolean, vsm, okapiBM25")
    # ranked là list các arXiv ID; cắt top-k
    return ranked[:topk]

def write_run_trec(run_path: Path, run_name: str, qid: str, ranked_ids: list):
    with run_path.open("a", encoding="utf-8") as f:
        for rank, docid in enumerate(ranked_ids, 1):
            # score có thể suy từ rank nếu mô hình không trả score; dùng 1.0/rank để hợp lệ format
            score = 1.0 / rank
            f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {run_name}\n")

def evaluate_runs(qrels_path: str, run_files: dict, k_prec: int = 10, k_rec: int = 100):
    qrels = read_trec_qrels(qrels_path)
    metrics = [P@k_prec, Recall@k_rec, MAP@k_rec, nDCG@k_prec, MRR@k_prec]
    results = {}
    for name, path in run_files.items():
        run = read_trec_run(path)
        vals = {}
        for m, qid, val in iter_calc(metrics, qrels, run):
            vals.setdefault(str(m), []).append(val)
        agg = {m: sum(v)/len(v) if v else 0.0 for m, v in vals.items()}
        results[name] = agg
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../datasets/arXiv_papers.json")
    parser.add_argument("--queries", default="queries.csv")
    parser.add_argument("--qrels", default="qrels.txt")
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    run_paths = {
        "boolean": Path(args.outdir) / "boolean.trec",
        "vsm": Path(args.outdir) / "vsm.trec",
        "okapiBM25": Path(args.outdir) / "bm25.trec",
    }
    # clear cũ
    for p in run_paths.values():
        if p.exists(): p.unlink()

    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH", args.dataset)

    se, papers = load_engine(dataset_path)

    with open(args.queries, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row["qid"]).strip()
            q   = row["query"].strip()
            for algo, run_p in run_paths.items():
                ranked = get_ranked_results(se, papers, q, algo, args.topk)
                write_run_trec(run_p, algo, qid, ranked)

    scores = evaluate_runs(args.qrels, {
        "Boolean": str(run_paths["boolean"]),
        "VSM": str(run_paths["vsm"]),
        "BM25": str(run_paths["okapiBM25"]),
    })
    print("\n=== Offline Evaluation (macro-avg) ===")
    for name, agg in scores.items():
        print(f"\n{name}")
        for m, v in agg.items():
            print(f"  {m}: {v:.4f}")

if __name__ == "__main__":
    main()
