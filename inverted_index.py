import os
import shutil
import time
import math
from xml.etree import ElementTree
from collections import Counter

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import *
from whoosh.reading import TermNotFound
from whoosh.query import Every
from whoosh import scoring


#######################################################################################################################

# Customize parameters here:

corpus_dir = "../material/rcv1"      # Directory of your rcv1 folder
docs_to_index = 100                  # How many docs to add to index, set to None to add all of the docs in the corpus

#######################################################################################################################


# By default, the writer will have 1GB (1024 MB) as the maximum memory for the indexing pool
# However, the actual memory used will be higher than this value because of interpreter overhead (up to twice as much)
def indexing(corpus, ram_limit=1024):  # TODO: allow customization for text preprocessing using extra function arguments
    start_time = time.time()

    # TODO: split content according to XML tags
    schema = Schema(id=NUMERIC(stored=True), content=TEXT)

    # Clear existing temp/indexdir folder and make new one
    if os.path.exists(os.path.join("temp", "indexdir")):
        shutil.rmtree(os.path.join("temp", "indexdir"))
    os.makedirs(os.path.join("temp", "indexdir"))

    # Create index in temp/indexdir folder
    ix = create_in(os.path.join("temp", "indexdir"), schema)
    writer = ix.writer(limitmb=ram_limit)
    traverse_folders(writer, corpus)
    writer.commit()

    end_time = time.time()

    # Traverses all files in the indexdir folder to calculate disk space taken up by the index
    space = 0
    for subdir, dirs, files in os.walk(os.path.join("temp", "indexdir")):
        space += sum(os.stat(os.path.join("temp", "indexdir", file)).st_size for file in files)

    return ix, end_time - start_time, space


# Traverses all sub-folders/files in the corpus and adds every document to the index
def traverse_folders(writer, corpus, d_test=True):
    n_docs = 0
    for subdir, dirs, files in os.walk(corpus):
        for file in files:
            # Ignore non-document files
            if file != "MD5SUMS" and subdir != os.path.join(corpus, "codes") and subdir != os.path.join(corpus, "dtds"):
                # TODO: improve efficiency
                if d_test and subdir[-8:] < "19961001":  # If using d_test, ignore docs before 1996-10-01
                    continue
                doc, doc_id = extract_doc_content(os.path.join(subdir, file))
                writer.add_document(id=doc_id, content=doc)
                if (n_docs := n_docs + 1) == docs_to_index:
                    return


def extract_doc_content(file):
    tree = ElementTree.parse(file)
    root = tree.getroot()  # Root is <newsitem>
    res = root.attrib["date"] + " "
    doc_id = int(root.attrib["itemid"])   # The doc id is an attribute of the <newsitem> tag
    for child in root:
        if child.tag in ("headline", "dateline", "byline"):  # Just extract text
            res += child.text + " "
        elif child.tag == "text":  # Traverse all <p> tags and extract text from each one
            for paragraph in child:
                res += paragraph.text + " "
    return res, doc_id


def extract_topic_query(topic_id, index, k):
    topic_id = int(topic_id)-101       # Normalize topic identifier to start at 0
    with open(os.path.join(corpus_dir, "..", "topics.txt")) as f:
        topics = f.read().split("</top>")[:-1]

    norm_topics = normalize_topics(topics)

    topic = norm_topics[topic_id]

    schema = Schema(id=NUMERIC(stored=True), content=TEXT)

    # Delete directory if it already exists and create a new one
    if os.path.exists(os.path.join("temp", "topicindexdir")):
        shutil.rmtree(os.path.join("temp", "topicindexdir"))
    os.makedirs(os.path.join("temp", "topicindexdir"))

    # Create auxiliary index with only 1 "document" (in reality, a topic)
    aux_index = create_in(os.path.join("temp", "topicindexdir"), schema)
    writer = aux_index.writer()
    writer.add_document(id=0, content=topic)
    writer.commit()

    with aux_index.searcher() as aux_searcher:
        # Dictionary of term frequencies in the TOPIC
        tf_dic = {word.decode("utf-8"): aux_searcher.frequency("content", word)
                  for word in aux_searcher.lexicon("content")}
        n_tokens_in_topic = sum(tf_dic.values())
        tf_dic = {word: freq/n_tokens_in_topic for word, freq in tf_dic.items()}

        with index.searcher() as searcher:
            # Dictionary of document frequencies of each term against the DOCUMENT INDEX
            results = searcher.search(Every(), limit=None)  # Returns every document
            n_docs = len(results)
            df_dic = {word.decode("utf-8"): searcher.doc_frequency("content", word)
                      for word in aux_searcher.lexicon("content")}
            idf_dic = {word: math.log10(n_docs/(df+1)) for word, df in df_dic.items()}

    schema = Schema(id=NUMERIC(stored=True), content=TEXT)

    # Delete directory if it already exists and create a new one
    if os.path.exists(os.path.join("temp", "topicsindexdir")):
        shutil.rmtree(os.path.join("temp", "topicsindexdir"))
    os.makedirs(os.path.join("temp", "topicsindexdir"))

    # Create auxiliary index with the topics
    topic_index = create_in(os.path.join("temp", "topicsindexdir"), schema)
    writer = topic_index.writer()
    for i, topic in enumerate(norm_topics):
        writer.add_document(id=i+101, content=topic)
    writer.commit()

    with topic_index.searcher(weighting=scoring.TF_IDF()) as topic_searcher:
        topic_df_dic = {word.decode("utf-8"): topic_searcher.doc_frequency("content", word)
                        for word in topic_searcher.lexicon("content")}
        topic_idf_dic = {word: math.log10(100 / (df + 1)) for word, df in topic_df_dic.items()}

    # Variation of TF-IDF, that uses topic tf and topics idf but also the idf against the corpus
    tfidfs = {key: tf_dic[key] * idf_dic[key] * topic_idf_dic[key] for key in tf_dic}

    return list(tup[0] for tup in Counter(tfidfs).most_common(k))


def normalize_topics(topics):
    norm_topics = []
    for topic in topics:
        norm_topic = re.sub("<num> Number: R[0-9][0-9][0-9]", "", topic)
        for tag in ("<top>", "<title>", "<desc> Description:", "<narr> Narrative:"):
            norm_topic = norm_topic.replace(tag, "")
        norm_topics.append(norm_topic)
    return norm_topics


def boolean_query(topic, k, index):
    words = extract_topic_query(topic, index, k)
    with index.searcher() as searcher:
        # Retrieve every document id
        results = searcher.search(Every(), limit=None)
        # Initialize dictionary that counts how many query terms each document contains
        occurrences = {r["id"]: 0 for r in results}
        doc_ids = [r["id"] for r in results]
        doc_ids.sort()
        for word in words:
            try:
                for doc_id in searcher.postings("content", word).all_ids():
                    occurrences[doc_ids[doc_id]] += 1
            except TermNotFound:
                pass

        res = [doc_id for doc_id, occurrence in occurrences.items() if occurrence >= k - round(0.2*k)]
        res.sort()
        return res


def ranking(topic_id, p, index, model="TF-IDF"):
    topic_id = int(topic_id)-101       # Normalize topic identifier to start at 0
    if model == "TF-IDF":
        weighting = scoring.TF_IDF()
    elif model == "BM25":
        weighting = scoring.BM25F()
    else:
        raise ValueError("Invalid scoring model: please use 'TF-IDF' or 'BM25'")
    with open(os.path.join(corpus_dir, "..", "topics.txt")) as f:
        topics = f.read().split("</top>")[:-1]
    norm_topics = normalize_topics(topics)
    topic = norm_topics[topic_id]
    with index.searcher(weighting=weighting) as searcher:
        q = QueryParser("content", index.schema, group=OrGroup).parse(topic)
        results = searcher.search(q, limit=p)
        return [(r["id"], round(r.score, 4)) for r in results]


# Prints the entire index for debugging and manual analysis purposes
def print_index(index):
    with index.searcher() as searcher:
        results = searcher.search(Every(), limit=None)
        doc_ids = [r["id"] for r in results]
        doc_ids.sort()
        for word in searcher.lexicon("content"):
            print(word.decode("utf-8") + ": ", end="")
            for doc in searcher.postings("content", word).all_ids():
                print(doc_ids[doc], end=" ")
            print()


def convert_filesize(size):
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    i = 0
    while size // 1024 > 0 and i < 4:
        size /= 1024.0
        i += 1
    return str(round(size, 3)) + " " + suffixes[i]


def main():
    ix, ix_time, ix_space = indexing(corpus_dir)
    # print("Whole index:"); print_index(ix)
    print(f"Time to build index: {round(ix_time, 3)}s")
    print(f"Disk space taken up by the index: {convert_filesize(ix_space)}")

    with ix.searcher() as searcher:
        results = searcher.search(Every(), limit=None)  # Returns every document
        print(f"Number of indexed docs: {len(results)}")

    print("Boolean queries for topic 102 (k=3, 1 mismatch):")
    print(boolean_query(102, 3, ix))

    print("Ranked query (using BM25) for topic 101 (p=20):")
    print(ranking(101, 20, ix, "BM25"))


if __name__ == "__main__":
    main()
