# 🧠 NeuroQueue: AI-Driven Semantic Message Queue

**NeuroQueue** is a next-generation intelligent message broker that uses **Generative AI (Llama-3 via Groq)** and **Vector Memory (ChromaDB)** to route messages based on *semantic understanding*, not just keywords.

Unlike traditional DLQs (Dead Letter Queues) that rely on error codes, NeuroQueue "reads" the message content, understands the urgency, and makes decisions like a human operator.

![NeuroQueue Dashboard](https://via.placeholder.com/800x400?text=NeuroQueue+AI+Dashboard)

---
<img width="1918" height="895" alt="image" src="https://github.com/user-attachments/assets/49479bc6-afb9-478e-90ae-66f74c488147" />

## 🌟 Key Features

*   **🧠 Cortex Engine (LLM)**: Uses **Llama-3.3-70b** to analyze message *Intent*, *Priority*, and *Risk*.
    *   *Example*: "The server room is overly hot" -> **Priority: Critical** (even without the word "critical").
*   **🔍 Semantic Memory**: Stores every message as a vector in **ChromaDB**.
    *   Allows you to search: *"Find all infrastructure failures"* and get relevant results instantly.
*   **⚡ Smart Routing**:
    *   **CRITICAL_QUEUE**: High priority (> 0.7).
    *   **PAYMENT_QUEUE**: Financial transactions.
    *   **DLQ_INVESTIGATION**: High risk (> 0.8) or anomalies.
    *   **NORMAL_QUEUE**: Standard traffic.
*   **📂 Local Architecture**:
    *   **No Redis/Kafka required**: Uses a high-performance local JSONL stream broker.
    *   **100% Python**: Easy to deploy and modify.

---

## 🛠️ Architecture

```mermaid
graph LR
    P[Producer API] -->|POST /process| A[API Gateway]
    A --> G[Decision Graph]
    G -->|Embed| E[ETL Pipeline]
    E -->|Analyze| L[LLM Cortex (Groq)]
    E -->|Retrieve/Store| C[(ChromaDB)]
    L -->|Decide| R[Router]
    R -->|Log| M[(MongoDB)]
    R -->|Queue| Q[Local Broker]
```

---

## 🚀 Getting Started

### Prerequisites
*   **Python 3.10+**
*   **MongoDB** (Local or Atlas)
*   **Groq API Key** (for LLM capabilities)

### 1. Installation
Clone the repo and run the setup script:
```bash
./setup.sh
```
*This handles virtualenv creation, pip install, and .env generation.*

### 2. Configuration
Edit `.env` to add your keys:
```ini
GROQ_API_KEY=gsk_...
MONGO_URI=mongodb://localhost:27017
```

### 3. Run the System
Start the AI Node and Web UI:
```bash
./run.sh
```
*   **Dashboard**: [http://localhost:8000](http://localhost:8000)
*   **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 API Usage

Interact with NeuroQueue via REST API.

### Send a Message (Producer)
```bash
curl -X POST "http://localhost:8000/process" \
     -H "Content-Type: application/json" \
     -d '{"content": "Database is unresponsive!"}'
```

### Get History (Consumer)
```bash
curl "http://localhost:8000/history"
```

*For more details, see [Integration Guide](integration_guide.md).*

---

## 📂 Project Structure

*   `neuroqueue/`: Core package.
    *   `ai/`: LLM Service, Embedding Models, Torch Networks.
    *   `graph/`: LangGraph decision workflow.
    *   `api/`: FastAPI server & Web UI.
    *   `storage/`: MongoDB & ChromaDB interfaces.
    *   `queue/`: LocalBroker implementation.
*   `main.py`: Entry point.
*   `config/`: Settings & Logging.

---

## 🛡️ License
MIT License. Built for the Future of DevOps.
