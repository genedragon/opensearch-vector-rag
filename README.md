# advanced-rag-amazon-opensearch
Advanced RAG workshop with Amazon OpenSearch Service.

This is a workshop that demonstrate how Amazon OpenSearch Service and Amazon Bedrock service can be used to create a Retrieval augmented generation application architecture. The workshops contains 3 python notebooks each building a wine recommendation chatbot. 

Notebook 1 - Demonstrate how we can use AWS SDK to build RAG architecture by using Amazon Bedrock titan v2 model and Amazon OpenSearch service. In this workshop we use python SDK to call each of these services.

Notebook 2 - Demonstrate how you can leverage Neural search feature from opensearch to call Amazon Bedrock titan v2 model without having to call it through python SDK. This gives away the heavy lifting of converting text to vector embeddings in python code and leaves the developer to just write code for loading text data and call search with a text query only. Amazon Opensearch service will take care of the converting text to vectors. In this notebook we continue to use Anthropic Claude sonnet model from Amazon Bedrock through our AWS SDK code.

Notebook 3 - Demonstrate how you can use the Flow framework from OpenSearch to develop a complete RAG architecture with just a single API call. We create a workflow that contains all the steps including deploying connectors for Amazon Titan v2 and Anthropic claude model, we setup an index, and create a pipeline that takes care of converting text to vector at the time of ingest and query, and we deploy a RAG Tool that will help us chain all the steps in RAG architecture through a single agent execution. This shows how you can use opensearch to take away the heavy lifting and create quick chatbots over your knowledge bases without having to write much code in python.


You can launch python notebook to walk through each lab
