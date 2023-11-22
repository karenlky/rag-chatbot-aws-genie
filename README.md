## AWS Genie
AWS Genie is an intelligent RAG chatbot that uses Amazon Sagemaker, Amazon Bedrock and Langchain.

Amazon Sagemaker Jumpstart supports a pleothora of open-source models and has a partnership with HuggingFace, a popular open-source model hub. AWS Genie deploys a LLaMa2-13b model endpoint on Amazon Sagemaker Jumpstart for inference.

Amazon Bedrock provides API access to proprietary and open-source models. AWS Genie uses Amazon's in-house Titan LLM from Amazon Bedrock for embedding.

Langchain is an open-source library that has many functions to support GenAI application development and deployment, including document loaders and retrievers. AWS Genie uses Langchain to chain up all the components of the chatbot.

## Architecture used
<div align="center"><img src="images/Architecture-RAG.drawio.svg" alt="Architecture diagram"></div>

## QuickStart

Here is how you can easily get started using Genie

Checkout the code
```
git clone https://github.com/karenlky/rag-chatbot-aws-genie.git
cd rag-chatbot-aws-genie
```

There are three major code files in this repo:
* aws_langchain_rag_demo_redacted.ipynb: A detailed step-by-step guide that explains how Genie is built
* aws_langchain_rag_demo_redacted.py: Pure Python version of the Jupyter notebook, for the Streamlit frontend to call
* agent_aws_st.py: Streamlit frontend

Please note the #TODO items in the code files. You will have to finish these items before the files can be run.

Run the Agent
```
pip install -r requirements.txt
streamlit run agent_aws_st.py
# View at localhost:8501
```
To run Genie as a website, you can run the code in EC2 and attach an elastic IP to the instance.

## Example Outputs
Here are some example outputs

<div align="center"><img src="images/What_is_langchain.png" alt="Answer in context"></div>
<div align="center"><img src="images/What_is_the_best_snack_in_HK.png" alt="Answer not in context"></div>
