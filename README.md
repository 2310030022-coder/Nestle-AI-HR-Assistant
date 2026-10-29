
# Nestlé AI-Powered HR Assistant

## Project Overview
An AI-powered chatbot designed to answer questions about Nestlé's Human Resources Policy using document retrieval and natural language processing.

## Technologies Used
- Python
- Google Colab
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- FLAN-T5
- Gradio

## Features
- Extracts text from HR policy PDF documents
- Splits documents into smaller chunks
- Generates semantic embeddings
- Stores embeddings in ChromaDB
- Retrieves relevant HR policy information
- Generates answers using a language model
- Provides a conversational Gradio interface

## Project Workflow
1. Load the Nestlé HR policy PDF.
2. Split the document into text chunks.
3. Generate embeddings using Sentence Transformers.
4. Store document vectors in ChromaDB.
5. Retrieve relevant chunks based on user questions.
6. Generate answers using FLAN-T5.
7. Display responses through Gradio.

## How to Run
1. Open Nestle_AI_HR_Assistant.ipynb in Google Colab.
2. Upload the Nestlé HR policy PDF.
3. Install the required Python libraries.
4. Run the free embedding and ChromaDB cells.
5. Run the chatbot and launch Gradio.

## Limitations
- Responses depend on the information in the supplied PDF.
- The free language model may generate incomplete or inaccurate answers.
- The current implementation uses local embeddings and FLAN-T5 instead of the OpenAI models specified in the original assignment.
- The Gradio share link is temporary.

## Author
Aiswarya Panda
