from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import os,re
from langchain.schema import Document
from ragas.evaluation import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall
from datasets import Dataset
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')       # Model used to evaulate semantic similarity

# Initializing PDF Paths & hugging face token
token = "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"
pdf_paths = ["./pdfs/Leave_Policy.pdf", "./pdfs/Benefits_Policy.pdf", "./pdfs/HR_Guidelines.pdf"]

# Loading PDFs
def load_docs(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    return docs

# Splitting into chunks
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return split_docs

# Creating Vectorstore
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    return vectorstore

# Loading LLM - I am loading quantized versions and using device map "cuda" because I was running in Collab GPU runtime
def load_llm_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=token)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="cuda", use_auth_token=token)

    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.1, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=gen_pipeline)

# Creating QA Chain with most similar documents
def create_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


# RAGAS Evaluation - It requires an LLM for evaluation (& currently compatible with OpenAI / Cloud based LLMs)
def evaluate_ragas_result(query, answer, source_documents, ground_truth):
    contexts = [doc.page_content for doc in source_documents]

    # Create a Hugging Face Dataset
    ragas_data = Dataset.from_list([{
        "question": query,
        "answer": answer,
        "retrieved_contexts": contexts,
        "reference": ground_truth
    }])

    # Run evaluation
    result = evaluate(
        ragas_data,
        metrics=[answer_relevancy, context_precision, context_recall]
    )

    return result

# As Ragas evaluation not working for me because of absence of OpenAI key, I am using this semantic similarity match for answer evaluation
def semantic_similarity(answer, ground_truth):
    embeddings = model.encode([answer, ground_truth], convert_to_tensor=True)
    answer_vec, ground_truth_vec = embeddings
    sim_gt = util.cos_sim(answer_vec, ground_truth_vec).item()

    return {"similarity_with_ground_truth": sim_gt}


# This is my main function
prompt_template = PromptTemplate(
    template="""
You are an AI assistant whose task is to answer queries based on HR policy documents.
Answer the following question based only on the context provided.
If the answer is not in the context, just say "NA".

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

documents = load_docs(pdf_paths)
chunks = split_docs(documents)
vectordb = create_vectorstore(chunks)
retriever = vectordb.as_retriever(search_type="similarity",search_kwargs={"k": 3})
print("\n Documents loaded successfully , Loading LLMs...")

model_id_1 = "mistralai/Mistral-7B-Instruct-v0.3"                  # I am using Mistral model as my first LLM
llm_1 = load_llm_pipeline(model_id_1)

model_id_2 = "meta-llama/Meta-Llama-3-8B-Instruct"                 # I am using Llama model as my second LLM
llm_2 = load_llm_pipeline(model_id_1)


qa_1 = RetrievalQA.from_chain_type(
    llm=llm_1,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)
qa_2 = RetrievalQA.from_chain_type(
    llm=llm_2,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
)

while True:
    query = input("\n Please enter your question or type 'exit' to stop the session : ")      # Continuing to take inputs till user press 'exit'
    if query.lower() == "exit":
        break
    print("\n Answering with Mistral Model ")
    result_mistral = qa_1.invoke(query)
    raw_output = result_mistral['result']
    match = re.search(r"Answer:\s*(.+)", raw_output, re.DOTALL)   # It is returning answer along with prompt so filtering just the answer part
    if match:
        mistral_answer = match.group(1).strip()
    else:
        mistral_answer = raw_output.strip()
    print("\n Answer is :----   ", mistral_answer)
    ground_truth = input("\n For evaluation, please enter the ground truth :- ")  # Taking ground truth as input from user for evaluation of results
    try:
        eval_1 = evaluate_ragas_result(query, mistral_answer, result_mistral['source_documents'],ground_truth)
        print("\n For Mistral Model :- Ragas metrics - ",eval_1)
    except Exception as e:
        print("\n For Mistral Model :- Error in RAGAS Evaulation Metrics :- ",e)
        print("\n For Mistral Model :-  Evalution results via semantic similarity match  ")
        print(semantic_similarity(mistral_answer, ground_truth))              

    print("\n Answering with Llama Model ")
    result_llama = qa_2.invoke(query)
    raw_output = result_llama['result']
    match = re.search(r"Answer :\s*(\.+)", raw_output, re.DOTALL)   # It is returning answer along with prompt so filtering just the answer
    if match:
        llama_answer = match.group(1).strip()
    else:
        llama_answer = raw_output.strip()
    print("\n Answer is :----   ", llama_answer)

    try:
        print("\n RAGAS Evaluation Metrcis for Llama Model :------------")
        eval_2 = evaluate_ragas_result(query, llama_answer, result_llama['source_documents'],ground_truth)
        print("\n For Llama Model :- Ragas metrics - ",eval_2)
    except Exception as e:
        print("\n For Llama Model :- Error in RAGAS Evaulation Metrics :- ",e)
        print("\n For Llama Model :-  Evalution results via semantic similarity match  ")
        print(semantic_similarity(llama_answer, ground_truth))
