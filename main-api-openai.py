#with api call and engineer prompt

import os
os.environ["OPENAI_API_KEY"] = " "
from flask import Flask,request, jsonify
import langchain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
loader = DirectoryLoader('/Users/user1/Downloads/Antier-Sol/5ire/content/DB', glob="./*.txt", loader_cls=TextLoader)
documents = loader.load()

# splitting the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

print(len(texts))
print(texts[3])

persist_directory = 'db'

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

print(retriever.search_type)
print(retriever.search_kwargs)

def calculate_similarity(query, response):
    vectorizer = TfidfVectorizer()
    tfidf_query = vectorizer.fit_transform([query])
    tfidf_response = vectorizer.transform([response])
    similarity = cosine_similarity(tfidf_query, tfidf_response)
    return similarity[0][0]

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)

# Define the engineer prompt
engineer_prompt = ""

# Full example
@app.route('/ask', methods=['GET', 'POST'])
def ask():
    query = ""
    
    query = request.json['question']
    
    # Append the engineer prompt to the query
    query1 = engineer_prompt + " " + query
    
    llm_response = qa_chain(query1)
    similarity = calculate_similarity(query, llm_response['result'])
    status = 200
    if similarity < 0.03:
        status = 204
    


    # Remove the engineer prompt from the answer
    answer = llm_response['result'].replace(engineer_prompt, "")
    if query=="hi" or query=="hello" or query=="hey" or query=="han" or query=="namaste" or query=="hay":
        answer ="Hi, This is your TataBot, How can I assist you??"
        status=200
        similarity=1


    if query=="bye" or query=="see you" or query=="bye bye":
        answer ="Bye, it was nice talking to you"
        status=200
        similarity=1


    return jsonify({
        'answer': answer,
        'similarity': similarity,
        'status': status
    })


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True, port=6000)
