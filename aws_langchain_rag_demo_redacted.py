import langchain
import os
from sagemaker.jumpstart.model import JumpStartModel
import boto3
import json
from langchain.llms.sagemaker_endpoint import LLMContentHandler, SagemakerEndpoint
from langchain.embeddings import BedrockEmbeddings
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import RequestsHttpConnection
from langchain.chains import RetrievalQA
import sys
import re

langchain.debug=False

AWS_REGION = 'us-east-1'
AWS_REGION_AOS = "ap-southeast-1"
SAGEMAKER_LLM_MODEL_NAME = 'jumpstart-dft-meta-textgeneration-llama-2-13b-f' # TODO: Deploy Sagemaker endpoint model from console, then define Sagemaker endpoint model name here
CRAWLED_TXT_FILE_NAME = '' # TODO: store the list of URLs you want to turn into embeddings into a txt file and store the name here
OPENSEARCH_URL = "https://search-vector-db-for-search-ho5ekfvpkfhtqnkam2naxqugqe.ap-southeast-1.es.amazonaws.com"
OPENSEARCH_VECTOR_INDEX_HTML = "rag-demo-aws-genai-html-asia"

os.environ['AWS_DEFAULT_REGION']=AWS_REGION
# TODO: access as a IAM user. Ensure user has the appropriate permissions and store the access keys here
os.environ['AWS_ACCESS_KEY_ID']=''
os.environ['AWS_SECRET_ACCESS_KEY']=''

# TODO: store credentials for Amazon OpenSearch cluster ("login", "password") here
OPENSEARCH_http_auth=("", "") 

def init_llm_sm_endpoint():
    endpoint_name = SAGEMAKER_LLM_MODEL_NAME
    aws_region=AWS_REGION
    parameters = {"max_new_tokens": 1000, "temperature": 0.1}

    class ContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        # LLAMA-2 chat
        # note: the input format requirement differs from LLM to LLM
        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            input_str = json.dumps({"inputs" : [[{"role" : "system",
            "content" : "You are QnA bot to answer the questions based on the context. If it is not in the context, start your reply with \"Based on context, I cannot answer\""},
            {"role" : "user", "content" : prompt}]],
            "parameters" : {**model_kwargs}})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            response_json = json.loads(output.read().decode("utf-8"))
            return response_json[0]["generation"]["content"]
        
    content_handler = ContentHandler()

    sm_llm = SagemakerEndpoint(
        endpoint_name=endpoint_name,
        region_name=aws_region,
        model_kwargs=parameters,
        content_handler=content_handler,
        endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    )
    return sm_llm


def init_eb_bedrock():
    bedrock_client = boto3.client('bedrock-runtime',
                                region_name=AWS_REGION,
                                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    # TODO: allow Bedrock Titan access on the AWS console, and store model ID here
    modelId = "amazon.titan-embed-text-v1"
    bedrock_embeddings = BedrockEmbeddings(
        client=bedrock_client,
        region_name=AWS_REGION,
        model_id=modelId, 
    )

    return bedrock_embeddings


def prepare_html():
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
        )
    with open(CRAWLED_TXT_FILE_NAME) as f:
        urls = f.read().splitlines() 
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    contents_transformed, metadatas = [], []
    for doc_transformed in docs_transformed:
        content_transformed = text_splitter.split_text(doc_transformed.page_content)
        contents_transformed.extend(content_transformed)
        metadatas.extend([doc_transformed.metadata]*len(content_transformed))
    return contents_transformed, metadatas


# only run this once in the beginning to initialise the vector db and store the document embeddings into vector db
def store_into_vector_db():
    tmp  = prepare_html()
    contents_transformed = tmp[0]
    metadatas = tmp[1]
    vectorstore = OpenSearchVectorSearch.from_texts(
        contents_transformed,
        init_eb_bedrock(), # embedding model
        metadatas,
        opensearch_url=OPENSEARCH_URL,
        http_auth=OPENSEARCH_http_auth,
        timeout = 600,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=OPENSEARCH_VECTOR_INDEX_HTML,
        engine="faiss",
        bulk_size=10000
    )
    return vectorstore


def get_vector_store():
    vectorstore = OpenSearchVectorSearch(
        opensearch_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_VECTOR_INDEX_HTML,
        embedding_function=init_eb_bedrock(),
        http_auth=OPENSEARCH_http_auth,
        timeout = 600,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        engine="faiss"
    )
    return vectorstore


# helper method for streamlit, so that sources will not be displayed if not covered in repo
def check_no_source(answer):
    if re.search("I cannot answer\\S*", answer.strip()) is None:
        return False
    else:
        return True


# helper method for streamlit
def find_metadata_sources_from_documents(documents):
    source_list = []
    for document in documents:
        if 'source' in document.metadata:
            source = document.metadata['source']
            source_list.append(source)
    source_list = list(dict.fromkeys(source_list))
    return source_list


def query(query):
    llm = init_llm_sm_endpoint()
    retreiver = get_vector_store().as_retriever(search_type="similarity", search_kwargs={'k': 6, 'score_threshold': 0.8})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        retriever=retreiver,
                                        return_source_documents=True)
    return qa_chain({"query":query})

if __name__ == '__main__':
    query(sys.argv[0])