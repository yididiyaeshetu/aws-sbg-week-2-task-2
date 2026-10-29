# Week 2: Mini Semantic Search

## Project Overview
A semantic search system for FAQs using Python, Pandas, Sentence Transformers, and ChromaDB.

## Objectives
- Load FAQ data from CSV using Pandas.
- Clean and prepare FAQ text.
- Create text embeddings.
- Store embeddings in ChromaDB.
- Search and print the top 3 matches.

## Technologies
- Python
- Pandas
- Sentence Transformers
- ChromaDB
- Google Colab

## How to Run
Install the dependencies:

    python -m pip install -r requirements.txt

Run the Python program:

    python task.py

## Workflow
1. Load the FAQ CSV.
2. Clean the data.
3. Prepare text chunks.
4. Generate embeddings.
5. Store the embeddings in ChromaDB.
6. Search for the top 3 matches.

## Sample Query
I forgot my password and cannot access my account.

## Note
The sample FAQ dataset is for demonstration. The ChromaDB collection is stored in memory.
