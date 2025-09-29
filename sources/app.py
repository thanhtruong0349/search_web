from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
from SearchEngine import SearchEngine
from file_operations import retrieve_data

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

load_dotenv()
host = os.getenv("FLASK_HOST", "127.0.0.1")
port = int(os.getenv("FLASK_PORT", 5001))
debug = os.getenv("FLASK_DEBUG", "True").lower() in ['true', '1']
dataset_path = os.getenv("DATASET_PATH", "../datasets/arXiv_papers_less.json")

papers_collection = retrieve_data(dataset_path)
search_engine = SearchEngine()
search_engine.build_preprocessed_documents(papers_collection)
search_engine.build_inverted_index()
search_engine.init_query_processor(device="cpu")   # <-- Khởi tạo 1 lần duy nhất

def normalize_id(arxiv_id: str):
    """Chuẩn hóa arXiv ID để so sánh an toàn"""
    return arxiv_id.strip().lower().replace("v1", "").replace("v2", "")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        algorithm = request.form['algorithm']
        filter_criteria = request.form['filter_criteria']

        boolean_results, vsm_results, okapiBM25_results, bert_results = search_engine.search(query)

        if algorithm == 'boolean':
            results = boolean_results
        elif algorithm == 'vsm':
            results = vsm_results
        elif algorithm == 'okapiBM25':
            results = okapiBM25_results
        elif algorithm == 'bert':
            results = bert_results
        else:
            return render_template('search_form.html', error_message='Invalid retrieval algorithm.')

        if results:
            if algorithm == 'boolean':
                results_ranked = search_engine.rank_results(results, query)
            # bóc doc_id nếu kết quả có dạng (doc_id, score)
            # VSM / BM25: bóc doc_id từ (doc_id, score)
            elif algorithm in ['vsm', 'okapiBM25']:
                results_ranked = results

            # BERT: giữ nguyên (đã là list doc_id)
            elif algorithm == 'bert':
                results_ranked = results
                
            # --- nếu có filter ---
            if filter_criteria != 'none':
                filters = {filter_criteria: query}
                filtered_results = search_engine.filter_results(results_ranked, filters, papers_collection)
                if filtered_results:
                    return render_template('results.html', query=query, papers=filtered_results, num_results=len(filtered_results), algorithm=algorithm)
                else:
                    return render_template('results.html', query=query, no_results=True, num_results=0, algorithm=algorithm)
            else:
                result_papers = []
                for arxiv_id in results_ranked:
                    paper = next((p for p in papers_collection if p['arXiv ID'] == arxiv_id), None)
                    if paper:
                        result_papers.append(paper)
                return render_template('results.html', query=query, papers=result_papers, num_results=len(result_papers), algorithm=algorithm)
        else:
            return render_template('results.html', query=query, no_results=True, num_results=0, algorithm=algorithm)

    return render_template('search_form.html')

if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)
