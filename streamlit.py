import requests
import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up Langchain components (same as in your script)
os.environ["OPENAI_API_KEY"] = ""
loader = DirectoryLoader(
    '/Users/arshitarora/projects/custom_alchemy/content', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
embedding = OpenAIEmbeddings()
persist_directory = 'db'
vectordb = Chroma.from_documents(
    documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
retriever = vectordb.as_retriever()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(
), chain_type="stuff", retriever=retriever, return_source_documents=True)

def calculate_similarity(query, response):
    vectorizer = TfidfVectorizer()
    tfidf_query = vectorizer.fit_transform([query])
    tfidf_response = vectorizer.transform([response])
    similarity = cosine_similarity(tfidf_query, tfidf_response)
    return similarity[0][0]

def process_llm_response(query, llm_response):
    return llm_response['result']
    # You can also return similarity if needed

# Function to get cryptocurrency exchange rates
def get_exchange_rate(currency_code):
    endpoint = 'live'
    access_key = ''
    url = f'http://api.coinlayer.com/api/{endpoint}?access_key={access_key}'

    response = requests.get(url)

    if response.status_code == 200:
        exchange_rates = response.json()
        if currency_code in exchange_rates['rates']:
            rate = exchange_rates['rates'][currency_code]
            return f"{currency_code} Exchange Rate: {rate}"
        else:
            return "Currency code not found in exchange rates."
    else:
        return "API request was not successful."

# Streamlit UI
def main():
    st.title("5ire Assistant")

    query = st.text_input("Enter your question:")
    currency_code = st.text_input("Enter Currency Code (e.g., BTC):")

    if st.button("Ask"):
        full_query = " " + query
        llm_response = qa_chain(full_query)

        if currency_code:
            exchange_rate_response = get_exchange_rate(currency_code.upper())
            st.write(exchange_rate_response)
        else:
            llm_result = process_llm_response(query, llm_response)
            st.write(llm_result)

if __name__ == "__main__":
    main()
