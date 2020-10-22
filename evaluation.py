from whoosh.index import open_dir
from trectools import TrecQrel, TrecRun, TrecEval
from inverted_index import *
import matplotlib.pyplot as plt


def evaluation(topics, r_test, ix):
    # Recall-precision curves for different output sizes
    # MAP
    # BPREF
    # Cumulative gains
    # Efficiency

    # k = 3, p = 1000
    print("Executing boolean queries...")
    unranked_results = [boolean_query(topic, 3, ix) for topic in topics]
    print("Executing TF-IDF queries...")
    tfidf_results = [ranking(topic, 50000, ix, "TF-IDF") for topic in topics]
    print("Executing BM25 queries...")
    bm25_results = [ranking(topic, 50000, ix, "BM25") for topic in topics]

    # Query results are stored in temp/<scoring>/runs.txt, where scoring can either be "boolean", "tfidf" or "bm25"
    # Creating runs files for TrecTools
    print("Writing prep files...")
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
                if int(topic[1:]) in topics:
                    new.write(f"{topic[1:]} 0 {doc} {relevant}\n")

    # Judgment
    qrels_file = os.path.join("temp", "qrelstest.txt")
    qrels = TrecQrel(qrels_file)

    # Evaluation files are stored in temp/<scoring>/eval.csv, where scoring can either be "boolean", "tfidf" or "bm25"
    # Unranked evaluation
    runs_file = os.path.join("temp", "boolean", "runs.txt")
    print("Beginning evaluation for boolean retrieval.")
    evaluate(qrels, runs_file, topics, os.path.join("eval", "boolean.csv"))
    print("Done!")

    # TF-IDF evaluation
    runs_file = os.path.join("temp", "tfidf", "runs.txt")
    print("Beginning evaluation for TF-IDF retrieval.")
    evaluate(qrels, runs_file, topics, os.path.join("eval", "tfidf.csv"))
    print("Plotting Precision-Recall curves for each topic...")
    plot_rp_curve(qrels, topics, runs_file, tfidf_results)
    print("Done!")

    # BM25 evaluation
    runs_file = os.path.join("temp", "bm25", "runs.txt")
    print("Beginning evaluation for BM-25 retrieval.")
    evaluate(qrels, runs_file, topics, os.path.join("eval", "bm25.csv"))
    print("Plotting Precision-Recall curves for each topic...")
    plot_rp_curve(qrels, topics, runs_file, bm25_results)
    print("Done!")
    print("All evaluations finished. You can see detailed results in the 'eval' folder.")


def evaluate(qrels, runs_file, topics, path_to_csv):
    runs = TrecRun(runs_file)
    ev = TrecEval(runs, qrels)

    n_topics = len(topics)

    # Calculate various metrics for each query considering the runs/judgment files provided
    print("Calculating metrics...")
    res = ev.evaluate_all(per_query=True)

    # Write results of evaluation to csv file
    res.printresults(path_to_csv, "csv", perquery=True)

    # Calculate NDCG@100 for each query, since the previous metrics don't include it,
    # and append it to each line of the new csv file
    ndcgs = ev.get_ndcg(depth=100, per_query=True)
    values = [row['NDCG@100'] for i, row in ndcgs.iterrows()]   # Column name of Pandas dataframe storing the data
    with open(path_to_csv, 'r') as f:
        lines = [line[:-1] for line in f]              # Remove '\n' from the end of each line
        lines[0] += ",ndcg@100\n"                      # Add new column to header
        for i in range(1, n_topics+1):                 # Lines 1 to n contain metric values for each of the n queries
            lines[i] += "," + str(values[i-1]) + "\n"  # Line 1 (i) should store value 0 (i-1) - arrays start at 0
        global_ndcg = ev.get_ndcg(depth=100, per_query=False)     # Calculate global NDCG
        lines[n_topics+1] += "," + str(global_ndcg) + "\n"        # Append global NDCG to last line
    with open(path_to_csv, 'w') as f:
        f.writelines(lines)                            # Overwrite csv file with new content


def plot_rp_curve(qrels, topics, runs_file, results):
    runs = TrecRun(runs_file)
    ev = TrecEval(runs, qrels)

    new_qrels = ev.qrels.qrels_data.copy()
    relevant_docs = {topic: [] for topic in topics}
    for i, row in new_qrels.iterrows():
        if row["rel"] > 0:
            relevant_docs[row["query"]].append(row["docid"])

    for i, topic in enumerate(topics):
        precisions_aux = [0]
        recalls_aux = [0]
        for j in range(min(50001, len(results[i]))):
            if results[i][j][0] in relevant_docs[topic]:
                recalls_aux.append(recalls_aux[j] + 1)
                precisions_aux.append(precisions_aux[j] + 1)
            else:
                recalls_aux.append(recalls_aux[j])
                precisions_aux.append(precisions_aux[j])
        recalls = [x / ev.get_relevant_documents() for x in recalls_aux]
        precisions = [(x / i if i > 0 else 1) for i, x in enumerate(precisions_aux)]

        interpolated_precisions = precisions.copy()
        j = len(interpolated_precisions) - 2
        while j >= 0:
            if interpolated_precisions[j+1] > interpolated_precisions[j]:
                interpolated_precisions[j] = interpolated_precisions[j+1]
            j -= 1

        fig, ax = plt.subplots()
        for j in range(len(recalls)-2):
            ax.plot((recalls[j], recalls[j]), (interpolated_precisions[j], interpolated_precisions[j+1]),
                    'k-', label='', color='red')
            ax.plot((recalls[j], recalls[j+1]), (interpolated_precisions[j+1], interpolated_precisions[j+1]),
                    'k-', label='', color='red')
        ax.plot(recalls, precisions, 'k--', color='blue')
        ax.title.set_text("R"+str(topic))
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        fig.show()


def main():
    # This assumes you have already created the index
    # If you haven't, adjust the number of docs to index (10k+ recommended)
    # and the corpus directory in inverted_index.py and run it
    ix = open_dir(os.path.join("temp", "indexdir"))

    # 5 different topics with varying outcomes across the topics and the 3 kinds of system
    evaluation([104, 121, 138, 164, 185], os.path.join(corpus_dir, "..", "qrels.test"), ix)


if __name__ == "__main__":
    main()
