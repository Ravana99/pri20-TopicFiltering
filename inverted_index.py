import os
import shutil
import time
import math
from xml.etree import ElementTree
from collections import Counter

# from whoosh.index import open_dir
from whoosh.index import create_in
from whoosh.analysis import StemmingAnalyzer, SpaceSeparatedTokenizer, StandardAnalyzer
from whoosh.fields import *
from whoosh.qparser import *
from whoosh.reading import TermNotFound
from whoosh.query import Every
from whoosh import scoring


#######################################################################################################################

# Customize parameters here:

corpus_dir = os.path.join("..", "material", "rcv1")      # Directory of your rcv1 folder
docs_to_index = 10000                # How many docs to add to index, set to None to add all of the docs in the corpus
stemming = True

#######################################################################################################################


# By default, the writer will have 1GB (1024 MB) as the maximum memory for the indexing pool
# However, the actual memory used will be higher than this value because of interpreter overhead (up to twice as much)
def indexing(corpus, ram_limit=1024, d_test=True, stemmed=True):
    start_time = time.time()

    global stemming
    stemming = stemmed

    if stemming:
        analyzer = StemmingAnalyzer()
    else:
        analyzer = StandardAnalyzer()

    schema = Schema(doc_id=NUMERIC(stored=True),
                    date=TEXT(analyzer=SpaceSeparatedTokenizer()),
                    headline=TEXT(field_boost=1.5, analyzer=analyzer),
                    dateline=TEXT(analyzer=analyzer),
                    byline=TEXT(analyzer=analyzer),
                    content=TEXT(analyzer=analyzer))

    index_dir = os.path.join("indexes", "docs")

    # Clear existing indexes/docs folder and make new one
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)

    # Create index in indexes/docs folder
    ix = create_in(index_dir, schema)
    writer = ix.writer(limitmb=ram_limit)
    traverse_folders(writer, corpus, d_test=d_test)
    writer.commit()

    end_time = time.time()

    # Traverses all files in the indexes/docs folder to calculate disk space taken up by the index
    space = 0
    for subdir, dirs, files in os.walk(index_dir):
        space += sum(os.stat(os.path.join(index_dir, file)).st_size for file in files)

    return ix, end_time - start_time, space


# Traverses all sub-folders/files in the corpus and adds every document to the index
def traverse_folders(writer, corpus, d_test):
    n_docs = 0

    if d_test:
        subdirs = filter(lambda x: x >= "19961001" and x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus))
    else:
        subdirs = filter(lambda x: x not in ("codes", "dtds", "MD5SUMS"), os.listdir(corpus))

    for subdir in subdirs:
        for file in os.listdir(os.path.join(corpus, subdir)):
            doc_id, date, headline, dateline, byline, content = extract_doc_content(os.path.join(corpus, subdir, file))
            writer.add_document(doc_id=doc_id, date=date, headline=headline,
                                dateline=dateline, byline=byline, content=content)
            n_docs += 1
            if n_docs == docs_to_index:
                return


def extract_doc_content(file):
    tree = ElementTree.parse(file)
    root = tree.getroot()  # Root is <newsitem>
    doc_id = int(root.attrib["itemid"])   # The doc id is an attribute of the <newsitem> tag
    date = root.attrib["date"]
    headline, dateline, byline, content = "", "", "", ""
    for child in root:
        if child.tag == "headline":
            headline = (child.text if child.text is not None else "")
        elif child.tag == "dateline":
            dateline = (child.text if child.text is not None else "")
        elif child.tag == "byline":
            byline = (child.text if child.text is not None else "")
        elif child.tag == "text":  # Traverse all <p> tags and extract text from each one
            content = ""
            for paragraph in child:
                content += (paragraph.text if paragraph.text is not None else "") + "\n"
    return doc_id, date, headline, dateline, byline, content


def extract_topic_query(topic_id, index, k):
    topic_id = int(topic_id)-101  # Normalize topic identifier to start at 0
    with open(os.path.join(corpus_dir, "..", "topics.txt")) as f:
        topics = f.read().split("</top>")[:-1]

    norm_topics = remove_tags(topics)
    topic = norm_topics[topic_id]

    if stemming:
        schema = Schema(id=NUMERIC(stored=True), content=TEXT(analyzer=StemmingAnalyzer()))
    else:
        schema = Schema(id=NUMERIC(stored=True), content=TEXT())

    topic_index_dir = os.path.join("indexes", "aux_topic")

    # Delete directory if it already exists and create a new one
    if os.path.exists(topic_index_dir):
        shutil.rmtree(topic_index_dir)
    os.makedirs(topic_index_dir)

    # Create auxiliary index with only 1 "document" (in reality, a topic)
    aux_index = create_in(topic_index_dir, schema)
    writer = aux_index.writer()
    writer.add_document(id=0, content=topic)
    writer.commit()

    with aux_index.searcher() as aux_searcher:
        # Dictionary of term frequencies in the TOPIC
        tf_dic = {word.decode("utf-8"): aux_searcher.frequency("content", word)
                  for word in aux_searcher.lexicon("content")
                  if word.decode("utf-8") not in ("document", "relev", "irrelev", "relevant", "irrelevant")}
        n_tokens_in_topic = sum(tf_dic.values())
        tf_dic = {word: freq/n_tokens_in_topic for word, freq in tf_dic.items()}

        with index.searcher() as searcher:
            # Dictionary of document frequencies of each term against the DOCUMENT INDEX
            results = searcher.search(Every(), limit=None)  # Returns every document
            n_docs = len(results)
            df_dic = {word: sum([searcher.doc_frequency(field, word)
                      for field in ("date", "headline", "dateline", "byline", "content")])
                      for word in tf_dic}
            idf_dic = {word: math.log10(n_docs/(df+1)) for word, df in df_dic.items()}

    # Variation of TF-IDF, that uses topic tf and topics idf but also the idf against the corpus
    tfidfs = {key: tf_dic[key] * idf_dic[key] for key, value in df_dic.items() if value > 0}

    return list(tup[0] for tup in Counter(tfidfs).most_common(k))


def remove_tags(topics):
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
        occurrences = {r["doc_id"]: 0 for r in results}
        doc_ids = [r["doc_id"] for r in results]
        doc_ids.sort()
        for word in words:
            search_occurrences(searcher, occurrences, doc_ids, word)

        res = [doc_id for doc_id, occurrence in occurrences.items() if occurrence >= k - round(0.2*k)]
        res.sort()
        return res


def search_occurrences(searcher, occurrences, doc_ids, word):
    aux_occurrences = occurrences.copy()
    for field in ("date", "headline", "dateline", "byline", "content"):
        try:
            for doc_id in searcher.postings(field, word).all_ids():
                # Makes sure each entry is only incremented at the end once even if the term shows up in multiple fields
                occurrences[doc_ids[doc_id]] = aux_occurrences[doc_ids[doc_id]] + 1
        except TermNotFound:
            continue


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
    norm_topics = remove_tags(topics)
    topic = norm_topics[topic_id]

    if stemming:
        analyzer = StemmingAnalyzer()
    else:
        analyzer = StandardAnalyzer()

    tokens = [token.text for token in analyzer(topic)]
    string_query = ' '.join(tokens)
    with index.searcher(weighting=weighting) as searcher:
        q = MultifieldParser(("date", "headline", "dateline", "byline", "content"),
                             index.schema, group=OrGroup).parse(string_query)
        results = searcher.search(q, limit=p)
        return [(r["doc_id"], round(r.score, 4)) for r in results]


# Prints the entire index for debugging and manual analysis purposes
def print_index(index):
    with index.searcher() as searcher:
        results = searcher.search(Every(), limit=None)
        doc_ids = [r["doc_id"] for r in results]
        doc_ids.sort()
        for field in ("date", "headline", "dateline", "byline", "content"):
            print(f"Index for field {field}:")
            for word in searcher.lexicon(field):
                print(word.decode("utf-8") + ": ", end="")
                for doc in searcher.postings(field, word).all_ids():
                    print(doc_ids[doc], end=" ")
                print()


def convert_filesize(size):
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    i = 0
    while size // 1024 > 0 and i < 4:
        size /= 1024.0
        i += 1
    return str(round(size, 3)) + " " + suffixes[i]
