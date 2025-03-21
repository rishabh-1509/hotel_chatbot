# Hotel Booking Analytics & QA System

This project implements a comprehensive system for analyzing hotel booking data and providing a natural language question-answering capability. The system uses a combination of data analysis, vector databases, and language models to extract insights and answer user queries.

## Features

1. **Data Processing & Analytics**
   - Revenue trends over time
   - Cancellation rate analysis
   - Geographical distribution of bookings
   - Booking lead time analysis
   - Length of stay analysis
   - Customer segmentation

2. **RAG-based Question Answering**
   - Uses ChromaDB for vector storage
   - Implements Retrieval-Augmented Generation (RAG)
   - Provides natural language responses to user queries

3. **REST API**
   - `/analytics` endpoint for getting insights
   - `/ask` endpoint for answering booking-related questions
   - `/health` endpoint for system status

## Setup Instructions

### Prerequisites
- Python 3.9+
- Docker (optional)

### Option 1: Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rishabh-1509/hotel_chatbot.git
   cd hotel-booking-analytics
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   ```bash
   mkdir -p data
   # Download hotel_bookings.csv from Kaggle and place it in the data directory
   ```

5. Run the application:
   ```bash
   python main.py
   ```

## API Usage

### Get Analytics Insights

```bash
curl -X POST http://localhost:8000/analytics
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Show me total revenue for July 2017"}'
```

### Check System Health

```bash
curl http://localhost:8000/health
```

## Sample Test Queries

1. "Show me total revenue for July 2017."
2. "Which locations had the highest booking cancellations?"
3. "What is the average price of a hotel booking?"
4. "What is the cancellation rate for city hotels?"
5. "Which market segment has the highest booking percentage?"

## Implementation Details

### Data Processing

The system uses pandas for data cleaning and analysis. It handles missing values, removes duplicates, and creates additional features like total revenue and total stays.

### Vector Database

ChromaDB is used to store and retrieve vector embeddings of the insights. This allows for semantic searching and retrieval of relevant information.

### Language Model

The system uses a Hugging Face model (FLAN-T5) for generating human-like responses to questions. The model is integrated with the vector database through a retrieval-augmented generation (RAG) approach.

### Performance

The system includes performance evaluation metrics such as response time and accuracy. These metrics are tracked and can be used to optimize the system.

## Future Improvements

1. Implement real-time data updates
2. Add query history tracking
3. Improve the RAG system with more sophisticated prompts
4. Add more visualization options
5. Implement user authentication and authorization

## License

This project is licensed under the MIT License - see the LICENSE file for details.
