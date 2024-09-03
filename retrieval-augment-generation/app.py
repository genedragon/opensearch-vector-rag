import boto3
import json
from tqdm import tqdm
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearch_py_ml.ml_commons import MLCommonClient
from requests_aws4auth import AWS4Auth
import requests 
from PIL import Image
import streamlit as st
import base64

## Setup variables to use for the rest of the app
cloudformation_stack_name = "multimodal-rag-opensearch"

# Create a Boto3 session
session = boto3.Session()
# Get the account id
account_id = boto3.client('sts').get_caller_identity().get('Account')
# Get the current region
region = session.region_name

# Get connector IDs from previous workshop steps
with open('connector_ids.json', 'r') as file:
    connector_ids = json.load(file)

service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
headers = {"Content-Type": "application/json"}


temp_dir = "/home/ec2-user/SageMaker/advanced-rag-amazon-opensearch/retrieval-augment-generation/"

st.set_page_config(layout="wide", page_title='Vector search with Amazon OpenSearch Service')
#st.title('Vector search with Amazon OpenSearch Service')

cfn = boto3.client('cloudformation')

# Method to obtain output variables from Cloudformation stack. 
def get_cfn_outputs(stackname):
    outputs = {}
    for output in cfn.describe_stacks(StackName=stackname)['Stacks'][0]['Outputs']:
        outputs[output['OutputKey']] = output['OutputValue']
    return outputs

def image_search(image, text):
    with open(image, "rb") as image_file:
        query_image_binary = base64.b64encode(image_file.read()).decode("utf8")
        
    keyword_payload = {"_source": {
            "exclude": [
                "vector_embedding"
            ]
            },
            "query": {    

            "neural": {
                "vector_embedding": {

                "query_image":query_image_binary,
                "query_text":text,

                "model_id": connector_ids['embedding_model_id'],
                "k": 5
                }

                }
                        }

            ,"size":5,
      }
    r = requests.get(url, auth=awsauth, json=keyword_payload, headers=headers)
    response_ = json.loads(r.text)
    
    return response_['hits']['hits']

outputs = get_cfn_outputs(cloudformation_stack_name)
aos_host = outputs['OpenSearchDomainEndpoint']
s3_bucket = outputs['s3BucketTraining']
bedrock_inf_iam_role = outputs['BedrockBatchInferenceRole']
bedrock_inf_iam_role_arn = outputs['BedrockBatchInferenceRoleArn']
sagemaker_notebook_url = outputs['SageMakerNotebookURL']

## Create a connection with OpenSearch domain.
# Retrieving credentials from Secrets manager¶
kms = boto3.client('secretsmanager')
aos_credentials = json.loads(kms.get_secret_value(SecretId=outputs['OpenSearchSecret'])['SecretString'])

#credentials = boto3.Session().get_credentials()
#auth = AWSV4SignerAuth(credentials, region)
auth = (aos_credentials['username'], aos_credentials['password'])

aos_client = OpenSearch(
    hosts = [{'host': aos_host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)
ml_client = MLCommonClient(aos_client)

st.header("Vector search with Amazon OpenSearch Service")

tab1, tab2, tab3 = st.tabs(["Lexical Search", "Multimodal Search", "Conversational Search"])

with tab1:
    st.header("Lexical Search")
    # st.write(connector_ids)
with tab2:
    st.header("Multimodal Search")
with tab3:
    st.header("Conversational Search")

# Tab1    
# Text search
# tab1.header("Lexical Search")
query_1 = tab1.text_input("Keywords",placeholder="Type your search here.")
url_1 = 'https://' + aos_host + "/bedrock-multimodal-rag/_search"
keyword_payload_1 = {"_source": {
        "exclude": [
            "vector_embedding"
        ]
        },
        "query": {    "match": {
                        "product_description": {
                            "query": query_1
                        }
                        }
                    }
        ,"size":5,
  }

r_1 = requests.get(url_1, auth=awsauth, json=keyword_payload_1, headers=headers)
response_1 = json.loads(r_1.text)
docs_1 = response_1['hits']['hits']

for i,doc in enumerate(docs_1):
    tab1.write(str(i+1)+ ". "+doc["_source"]["product_description"])
    image = Image.open(temp_dir + doc["_source"]["image_url"])
    tab1.image(image)

# Tab2
# Multimodal Search
# Text and image as inputs
s3 = boto3.client('s3')
url = 'https://'+aos_host + "/bedrock-multimodal-rag/_search"
query_2 = tab2.text_input("Keywords", placeholder="Type your search text here.", value="colorful")

# Define the image URLs or file paths
image1 = "tmp/images/footwear/10.jpg"
image2 = "tmp/images/floral/17b69fda-f8a8-4523-9181-3e3e65887a97.jpg"
image3 = "tmp/images/homedecor/c6d7f153-e5a7-4168-a2f0-7471520e3f00.jpg"

# Create a list of image options
image_options = ["Image 1", "Image 2", "Image 3"]

col1, col2, col3 = tab2.columns(3)

response_mm = []

with col1:
    with st.form("Image1"):
        st.write("Image 1")
        st.image(image1, width=300)
        submitted = st.form_submit_button("Search with image 1")
        if submitted:
             response_mm = image_search(image1, query_2)
with col2:
    with st.form("Image2"):
        st.write("Image 2")
        st.image(image2, width=300)
        submitted = st.form_submit_button("Search with image 2")
        if submitted:
             response_mm = image_search(image2, query_2)
with col3:
    with st.form("Image3"):
        st.write("Image 3")
        st.image(image3, width=300)
        submitted = st.form_submit_button("Search with image 3")
        if submitted:
             response_mm = image_search(image3, query_2)
    
for i in response_mm:
    tab2.write(str(i["_source"]["product_description"]))
    image = Image.open(temp_dir + i["_source"]["image_url"])
    tab2.image(image)
    

# Tab3
# Conversational Search
def response_generator(prompt, memory_id):
    query = prompt
    url = 'https://' + aos_host + "/bedrock-multimodal-rag/_search?search_pipeline=multimodal_rag_pipeline"

    ## replace image
    img = Image.open("tmp/images/accessories/1.jpg") 
    print("Input query Image:")
    img.show()
    with open("tmp/images/accessories/1.jpg", "rb") as image_file:
        query_image_binary = base64.b64encode(image_file.read()).decode("utf8")

    multimodal_payload = {
        "_source": {
            "exclude": [
                "vector_embedding", "image_binary"
            ]},
        "query": {
            "neural": {
        "vector_embedding": {
            "query_image":query_image_binary,
            "query_text": query,
            "model_id": connector_ids['embedding_model_id'],
            "k": 5
        }
        }
            },
        "size":5,
        "ext": {
        "generative_qa_parameters": {
        "llm_model": "bedrock/claude",
        "llm_question": query,
        "memory_id": memory_id ,
        "context_size": 5,
        "message_size": 5,
        "timeout": 60
        }
    }
    }

    r = requests.get(url, auth=awsauth, json=multimodal_payload, headers=headers)
    response_ = json.loads(r.text)
    rag = response_['ext']['retrieval_augmented_generation']

    response = {}
    response['answer'] = rag['answer']
    
    return response

#     STREAM RESPONSE using yield
#     for word in response.split():
#         yield word + " "
#         time.sleep(0.05)
                        

# Display chat messages from history on app rerun
@st.fragment
def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

@st.fragment
def new_chat():
    ## Initialize memory id variable to use in the next conversational search
    memory_id= ''
    path = '/_plugins/_ml/memory/'
    url = 'https://'+aos_host + '/' + path
    payload = {   
        "name": "Conversation about bags"
    }
    r = requests.post(url, auth=awsauth, json=payload, headers=headers)
    memory_id = json.loads(r.text)["memory_id"]
    st.session_state.memory_id = memory_id
    ## Initialize chat history
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Please ask a question."})

if 'memory_id' not in st.session_state:
    new_chat()

with tab3.container(border=True):
    st.write("Memory ID: " + st.session_state.memory_id)
    st.button("New chat",on_click=new_chat)

with tab3:
    display_chat()
            
# Accept user input
if prompt := tab3.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with tab3.chat_message("user"):
        st.markdown(prompt)
    response = response_generator(prompt, st.session_state.memory_id)
    # Display assistant response in chat message container
    with tab3.chat_message("assistant"):
        st.markdown(response['answer'])        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
