---
title: SHL Assessment Recommender
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# SHL Assessment Recommender

An AI-powered conversational recommendation system that helps recruiters and hiring managers discover the most suitable **SHL Individual Test Solutions** through natural language conversations.

This project was developed as part of the **SHL AI Intern Take-Home Assignment**.

---


## Features

- Conversational assessment recommendation
- Semantic search using FAISS
- Gemini-powered grounded responses
- Context-aware conversation handling
- Clarification for vague queries
- Recommendation refinement
- Assessment comparison
- Prompt injection protection
- Off-topic request refusal
- Stateless FastAPI API
- SHL catalog-based recommendations only

---

## Tech Stack

- Python 3.11+
- FastAPI
- Google Gemini 2.5 Flash
- Sentence Transformers
- FAISS
- Pydantic
- Uvicorn

---

## Project Structure

```
.
├── app.py
├── chatbot.py
├── conversation.py
├── intent.py
├── retriever.py
├── search.py
├── prompt_builder.py
├── llm.py
├── vector_store.py
├── models.py
├── config.py
│
├── data/
│   └── catalog.json
│
├── embeddings/
│   ├── faiss.index
│   └── metadata.pkl
│
├── .env
├── requirements.txt
└── README.md
```

---

## Dataset

The recommender uses the **SHL Product Catalog (Individual Test Solutions)**.

Each assessment contains metadata including:

- Assessment Name
- Description
- Job Levels
- Languages
- Remote Testing
- Adaptive Testing
- Assessment Type
- Catalog URL

The catalog is converted into vector embeddings using Sentence Transformers and indexed with FAISS for semantic retrieval.

---

## Architecture

```
User

↓

FastAPI

↓

chatbot.py

↓

conversation.py

↓

intent.py

↓

retriever.py

↓

search.py (FAISS)

↓

Gemini

↓

Response
```

---

## Workflow

1. User sends conversation history.
2. Conversation constraints are extracted.
3. User intent is detected.
4. Semantic search retrieves relevant assessments.
5. Metadata-based reranking improves relevance.
6. Gemini generates a grounded response using only retrieved assessments.
7. Structured recommendations are returned.

---

## API Endpoints

### Health Check

```
GET /health
```

Response

```json
{
  "status": "ok"
}
```

---

### Chat Endpoint

```
POST /chat
```

Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java Developer with 4 years experience."
    }
  ]
}
```

Example Response

```json
{
  "reply": "Based on your requirements, here are suitable SHL assessments.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/...",
      "test_type": "Knowledge & Skills"
    }
  ],
  "end_of_conversation": true
}
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Generate Embeddings

Run

```bash
python vector_store.py
```

This generates

```
embeddings/faiss.index
embeddings/metadata.pkl
```

---

## Run the Application

```bash
uvicorn app:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

## Conversation Capabilities

The chatbot supports:

- Clarifying vague requests
- Context-aware recommendations
- Refining recommendations
- Comparing SHL assessments
- Handling changing user requirements
- Refusing unrelated questions
- Preventing prompt injection
- Returning only SHL catalog assessments

---

## Retrieval Pipeline

```
Conversation

↓

Constraint Extraction

↓

Semantic Search (FAISS)

↓

Metadata Reranking

↓

Gemini Response

↓

Recommendations
```

---

## Testing Scenarios

The system has been tested with:

- Vague hiring requests
- Technical hiring
- Personality assessment requests
- Recommendation refinement
- Assessment comparison
- Prompt injection attempts
- Off-topic queries
- Remote assessment filtering
- Adaptive assessment filtering

---

## Limitations

- Recommendations are limited to the provided SHL catalog.
- Results depend on semantic similarity and catalog metadata.
- Requires a valid Gemini API key.

---

## Future Improvements

- Hybrid keyword + semantic retrieval
- Learning-to-rank reranking
- Multilingual conversations
- Assessment explanation scoring
- Conversation memory summarization
- Feedback-based recommendation optimization

---

## Author

**Sahil Yadav**

Developed as part of the SHL AI Intern Take-Home Assignment.