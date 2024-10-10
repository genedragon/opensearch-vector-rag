import boto3
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
import os
import urllib.request
import tarfile
from requests_aws4auth import AWS4Auth
from ruamel.yaml import YAML
from PIL import Image
import base64
import re
import streamlit as st

with open('connector_ids.json', 'r') as file:
    connector_ids = json.load(file)
aos_host = connector_ids['aos_host']

# Create a Boto3 session
session = boto3.Session()
# Get the account id
account_id = boto3.client('sts').get_caller_identity().get('Account')
# Get the current region
region = session.region_name

# Connect to OpenSearch using the IAM Role of this Jupyter notebook
# Create AWS4Auth instance
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    'es',
    session_token=credentials.token
)

# Create OpenSearch client
aos_client = OpenSearch(
    hosts=[f'https://{aos_host}'],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

st.header("Shopping Assistant App with Amazon OpenSearch Service")

@st.fragment
def response_generator(prompt):
    # RAG using multimoadal search to provide prompt context
    # Text and image as inputs
    # query = "for hiking"
    query = prompt
    img = Image.open("./simple_bag.jpg") 
    print("Input text query: "+query)
    print("Input query Image:")
    img.show()

    # Define the query and search body
    with open("./simple_bag.jpg", "rb") as image_file:
        query_image_binary = base64.b64encode(image_file.read()).decode("utf8")

    response = aos_client.search(
        index='bedrock-multimodal-rag',
        body={
            "_source": {
                "exclude": ["vector_embedding", "image_binary"]
            },
            "query": {
                "neural": {
                    "vector_embedding": {
                        "query_image": query_image_binary,
                        "query_text": query,
                        "model_id": connector_ids['embedding_model_id'],
                        "k": 5
                    }
                }
            },
            "size": 5,
            "ext": {
                "generative_qa_parameters": {
                    "llm_model": "bedrock/claude",
                    "llm_question": query,
                    "memory_id": st.session_state.memory_id,
                    "context_size": 5,
                    "message_size": 5,
                    "timeout": 60
                }
            }
        },
        params={
            "search_pipeline": "multimodal_rag_pipeline"
        },
        request_timeout=30
    )
    # Extract the generated 'shopping assistant' recommendations
    # Split the string into lines
    lines = response['ext']['retrieval_augmented_generation']['answer'].split('\n')
    recommendations = []
    for line in lines:
        if re.match(r'[^\s\0]+', line):
            recommendations.append(line.strip())
    
    return response

#     STREAM RESPONSE using yield
    # for word in response.split():
    #     yield word + " "
    #     time.sleep(0.05)

def new_chat_memory_id():
    # Prepare the query string
    payload = {
        
        "name": "Conversation about products"
    }
    # Make the request
    response = aos_client.transport.perform_request(
        'POST',
        "/_plugins/_ml/memory/",
        body=payload,
        headers={"Content-Type": "application/json"}
    )
    # Persist memory_id
    st.session_state.memory_id = response['memory_id']
    st.session_state.messages = []

if 'memory_id' not in st.session_state:
    new_chat_memory_id()   

with st.form("memory_id_display"):
    st.write("Memory ID: " + st.session_state.memory_id)
    st.form_submit_button("New chat",on_click=new_chat_memory_id)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = response_generator(prompt)
    # Extract the generated 'shopping assistant' recommendations
    # Split the string into lines
    lines = response['ext']['retrieval_augmented_generation']['answer'].split('\n')
    recommendations = []
    for line in lines:
        if re.match(r'[^\s\0]+', line):
            recommendations.append(line.strip())

    # Output the search results and shopping assistnat recommendations together
    with st.chat_message("assistant"):
        st.markdown('Search results and Shopping assistant recommendations:')
        count = 1
        for hit in response['hits']['hits']:
            st.markdown("Search result "+str(count) + ": ")
            st.markdown(hit["_source"]["product_description"])
            st.markdown("Shopping assistant: ")
            st.markdown(recommendations[count-1])
            image = Image.open(hit["_source"]["image_url"])
            new_size = (300, 200)
            resized_img = image.resize(new_size)
            st.image(resized_img)
            count+=1
            st.markdown('')

    # # Display assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    # # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": response})