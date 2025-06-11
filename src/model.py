import os 
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
# from google.generativeai import GenerativeModel
# import google.generativeai as genai
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from dotenv import load_dotenv
load_dotenv()
def get_azure_openai_model():
    endpoint = os.getenv("ENDPOINT_URL")
    deployment = os.getenv("DEPLOYMENT_NAME")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("API_VERSION")



    model = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
    )
    return model

def get_azure_openai_chat_model():
    OPENAI_API_BASE=os.getenv("ENDPOINT_URL_chat")  #https://github.com/langchain-ai/langchain/issues/13284#issuecomment-1812873075
    OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION=os.getenv("API_VERSION")
    OPENAI_API_TYPE="azure"
    DEPLOYMENT_NAME_GPT_4="gpt-4o-mini-test"

    model = AzureChatOpenAI(
        openai_api_base= OPENAI_API_BASE,
        openai_api_key=OPENAI_API_KEY,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_type= OPENAI_API_TYPE,
        model_name=DEPLOYMENT_NAME_GPT_4,
        temperature=0.0
    )
    return model

def get_azure_openai_mini_model():
    OPENAI_API_BASE=os.getenv("ENDPOINT_URL") 
    OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION=os.getenv("API_VERSION_MINI")
    OPENAI_API_TYPE="azure"
    DEPLOYMENT_NAME_GPT_4="o1-mini"

    model = AzureOpenAI(
        azure_endpoint = OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        azure_deployment=DEPLOYMENT_NAME_GPT_4
    )
    return model

def get_llamaindex_model_mini():
    endpoint = "https://d-ais-eus-ais-chatbots.openai.azure.com/"
    model_name = "o1-mini"
    deployment = "o1-mini"
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = "2024-12-01-preview" # Use a valid API version

    llm = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
        deployment_name=deployment,
        model_name=model_name,
        temperature=1.0
    )
    return llm

def get_llamaindex_model():
    endpoint = "https://d-ais-eus-ais-chatbots.openai.azure.com/"
    model_name = "o1-mini"
    deployment = "o1-mini"
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = "2024-12-01-preview" # Use a valid API version

    llm = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version="2024-05-01-preview",
        deployment_name="gpt-4o-mini-test",
        model_name="gpt-4o-mini-test",
        temperature=0.0
    )
    return llm

def get_huggingface_embedding_model():
    embedding_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return embedding_model


# def get_gemini_model():
#     GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
#     genai.configure(api_key=GOOGLE_API_KEY)
#     model = GenerativeModel("gemini-2.0-pro-exp-02-05")  # Adjust model version as needed gemini-2.0-pro-exp-02-05 , gemini-2.0-flash
#     return model