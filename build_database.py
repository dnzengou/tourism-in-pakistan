from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


def main():
    loader = WebBaseLoader([
        "https://visitrwanda.com/",
        "https://en.wikipedia.org/wiki/Tourism_in_Pakistan",
        "https://traveltriangle.com/blog/places-to-visit-in-rwanda/",
        "https://www.investin-rwanda.com",
        "https://rdb.rw/investment-opportunities/invest-in-tourism/",
        "https://rdb.rw/rwandas-tourism-offerings-and-investment-opportunities-highlighted-at-iltm-asia-pacific-2023/"
        ])
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    # db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

if __name__ == "__main__":
    main()
