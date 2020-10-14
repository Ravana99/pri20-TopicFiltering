import os
import shutil
import time
from whoosh.index import create_in
from whoosh.fields import *

# from whoosh.qparser import *       # to actually process queries
# from whoosh.index import open_dir  # to open already existing index from folder
# from whoosh import query           # to query for every document
# from whoosh import scoring         # to use different scoring
from whoosh.query import Every       # for testing, to retrieve every document


#######################################################################################################################

# Customize parameters here:

corpus_dir = "../material/rcv1"     # Directory of your rcv1 folder
docs_to_index = 250                 # How many docs to add to index, set to None to add all of the docs in the corpus

#######################################################################################################################


def indexing(corpus):  # TODO: allow customization for text preprocessing using extra function arguments
    start_time = time.time()

    # TODO: split content according to XML tags
    schema = Schema(id=NUMERIC(stored=True), content=TEXT)

    # Clear existing indexdir folder and make new one
    if os.path.exists("indexdir"):
        shutil.rmtree("indexdir")
    os.makedirs("indexdir")

    # Create index in indexdir folder
    ix = create_in("indexdir", schema)
    writer = ix.writer()
    traverse_folders(writer, corpus)
    writer.commit()

    end_time = time.time()

    # Traverses all files in the indexdir folder to calculate disk space taken up by the index
    space = 0
    for subdir, dirs, files in os.walk("indexdir"):
        space += sum(os.stat(os.path.join("indexdir", file)).st_size for file in files)

    return ix, end_time - start_time, space


# Traverses all sub-folders/files in the corpus and adds every document to the index
def traverse_folders(writer, corpus):
    n_docs = 0
    for subdir, dirs, files in os.walk(corpus):
        for file in files:
            # Ignore non-document files
            if file != "MD5SUMS" and subdir != os.path.join(corpus, "codes") and subdir != os.path.join(corpus, "dtds"):
                with open(os.path.join(subdir, file), "r") as f:
                    writer.add_document(id=file[:-10], content=f.read())
                if (n_docs := n_docs + 1) == docs_to_index:
                    return


def convert_filesize(size):
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    i = 0
    while size // 1024 > 0 and i < 4:
        size /= 1024.0
        i += 1
    return str(round(size, 3)) + " " + suffixes[i]


def main():
    (ix, ix_time, ix_space) = indexing(corpus_dir)
    print(f"Time to build index: {round(ix_time, 3)}s")
    print(f"Disk space taken up by the index: {convert_filesize(ix_space)}")

    with ix.searcher() as searcher:
        results = searcher.search(Every(), limit=None)  # Returns every document
        print(f"Number of docs returned: {len(results)}")
        # print(f"Docs returned: { {r['id']: r['content'] for r in results} }")
        # print(f"Doc scores: { {r['id']: r.score for r in results} }")  # Scores are always 1.0 with an Every() query


main()
