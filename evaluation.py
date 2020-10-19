from whoosh.index import open_dir
from trectools import TrecQrel, TrecRun, TrecEval
from inverted_index import *


def evaluation(topics, r_test, ix):
    # Recall-precision curves for different output sizes
    # MAP
    # BPREF
    # Cumulative gains
    # Efficiency

    # k = 3, p = 1000
    unranked_results = [boolean_query(topic, 3, ix) for topic in topics]
    tfidf_results = [ranking(topic, 1000, ix, "TF-IDF") for topic in topics]
    bm25_results = [ranking(topic, 1000, ix, "BM25") for topic in topics]

    # Query results are stored in temp/<scoring>/runs.txt, where scoring can either be "boolean", "tfidf" or "bm25"
    # Creating runs files for TrecTools
    with open(os.path.join("temp", "boolean", "runs.txt"), "w") as f:
        for i, topic in enumerate(unranked_results):
            for j, r in enumerate(topic):
                f.write(f"{topics[i]} Q0 {r} {j+1} 1 booleanIR\n")
    with open(os.path.join("temp", "tfidf", "runs.txt"), "w") as f:
        for i, topic in enumerate(tfidf_results):
            for j, r in enumerate(topic):
                f.write(f"{topics[i]} Q0 {r[0]} {j+1} {r[1]} tfidfIR\n")
    with open(os.path.join("temp", "bm25", "runs.txt"), "w") as f:
        for i, topic in enumerate(bm25_results):
            for j, r in enumerate(topic):
                f.write(f"{topics[i]} Q0 {r[0]} {j+1} {r[1]} bm25IR\n")

    # Creating qrels file with the right format (at temp/qrelstest.txt)
    with open(os.path.join("temp", "qrelstest.txt"), "w") as new:
        with open(r_test, "r") as f:
            for line in f:
                topic, doc, relevant = line.split()
                new.write(f"{topic[1:]} 0 {doc} {relevant}\n")

    # Judgment
    qrels_file = os.path.join("temp", "qrelstest.txt")
    qrels = TrecQrel(qrels_file)

    # Evaluation files are stored in temp/<scoring>/eval.csv, where scoring can either be "boolean", "tfidf" or "bm25"
    # Unranked evaluation
    runs_file = os.path.join("temp", "boolean", "runs.txt")
    evaluate(qrels, runs_file, os.path.join("eval", "boolean.csv"))
    # TF-IDF evaluation
    runs_file = os.path.join("temp", "tfidf", "runs.txt")
    evaluate(qrels, runs_file, os.path.join("eval", "tfidf.csv"))
    # BM25 evaluation
    runs_file = os.path.join("temp", "bm25", "runs.txt")
    evaluate(qrels, runs_file, os.path.join("eval", "bm25.csv"))


def evaluate(qrels, runs_file, path_to_csv):
    runs = TrecRun(runs_file)
    ev = TrecEval(runs, qrels)

    # Calculate various metrics for each query considering the runs/judgment files provided
    res = ev.evaluate_all(per_query=True)

    # Write results of evaluation to csv file
    res.printresults(path_to_csv, "csv", perquery=True)

    # Calculate NDCG for each query, since the previous metrics don't include it,
    # and append it to each line of the new csv file
    ndcgs = ev.get_ndcg(per_query=True)
    values = [row['NDCG@1000'] for i, row in ndcgs.iterrows()]   # Column name of Pandas dataframe storing the data
    with open(path_to_csv, 'r') as f:
        lines = [line[:-1] for line in f]              # Remove '\n' from the end of each line
        lines[0] += ",ndcg@1000\n"                     # Add new column to header
        for i in range(1, 101):                        # Lines 1-100 contain metric values for each of the 100 queries
            lines[i] += "," + str(values[i-1]) + "\n"  # Line 1 (i) should store value 0 (i-1) - arrays start at 0
        global_ndcg = ev.get_ndcg(per_query=False)     # Calculate global NDCG
        lines[101] += "," + str(global_ndcg) + "\n"    # Append global NDCG to last line
    with open(path_to_csv, 'w') as f:
        f.writelines(lines)                            # Overwrite csv file with new content


def main():
    # This assumes you have already created the index
    # If you haven't, adjust the number of docs to index (10k+ recommended)
    # and the corpus directory in inverted_index.py and run it
    ix = open_dir(os.path.join("temp", "indexdir"))

    evaluation(range(101, 111), os.path.join(corpus_dir, "..", "qrels.test"), ix)


if __name__ == "__main__":
    main()
