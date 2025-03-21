# Hotel Booking Analytics and QA System
# This system provides analytics for hotel booking data and implements a RAG-based Q&A system

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Union

# For vector DB
import chromadb
from chromadb.utils import embedding_functions

# For the LLM-based Q&A
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document

# For API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import time

# Set up directories
DATA_DIR = "data"
OUTPUT_DIR = "output"
CHROMA_DB_DIR = "chroma_db"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Data Processing Class
class HotelBookingProcessor:
    def __init__(self, file_path: str = "data/hotel_bookings.csv"):
        """Initialize the processor with the path to the CSV file."""
        self.file_path = file_path
        self.df = None
        self.insights = {}
        
    def load_data(self):
        """Load the data from the CSV file."""
        self.df = pd.read_csv(self.file_path)
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the data."""
        # Handle missing values
        numerical_cols_with_missing = ['children', 'agent', 'company']
        for col in numerical_cols_with_missing:
            self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        # Handle missing values in categorical columns
        categorical_cols_with_missing = ['country']
        for col in categorical_cols_with_missing:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        # Remove duplicates
        self.df.drop_duplicates(inplace=True)
        
        # Convert dates and create new features
        self.df['reservation_status_date'] = pd.to_datetime(self.df['reservation_status_date'])
        self.df['total_stays'] = self.df['stays_in_weekend_nights'] + self.df['stays_in_week_nights']
        
        # Create 'arrival_date' feature
        self.df['arrival_date'] = pd.to_datetime(
            self.df['arrival_date_year'].astype(str) + '-' + 
            self.df['arrival_date_month'] + '-' + 
            self.df['arrival_date_day_of_month'].astype(str), 
            errors='coerce'
        )
        
        # Calculate total revenue
        self.df['total_revenue'] = self.df['adr'] * self.df['total_stays']
        
        # Create lead time bins
        self.df['lead_time_bin'] = pd.cut(
            self.df['lead_time'], 
            bins=[0, 30, 90, 180, float('inf')], 
            labels=['Short', 'Medium', 'Long', 'Very Long']
        )
        
        return self.df
    
    def calculate_insights(self):
        """Calculate various insights from the data."""
        insights = {}
        
        # 1. Revenue trends over time
        monthly_revenue = self.df.groupby(pd.Grouper(key='arrival_date', freq='M'))['total_revenue'].sum()
        insights['monthly_revenue'] = monthly_revenue.to_dict()
        
        # Convert datetime keys to strings for JSON serialization
        insights['monthly_revenue'] = {str(k): v for k, v in insights['monthly_revenue'].items()}
        
        # Revenue by hotel type
        revenue_by_hotel = self.df.groupby('hotel')['total_revenue'].mean().to_dict()
        insights['revenue_by_hotel'] = revenue_by_hotel
        
        # Revenue by room type
        revenue_by_room = self.df.groupby('reserved_room_type')['total_revenue'].mean().to_dict()
        insights['revenue_by_room'] = revenue_by_room
        
        # Revenue by market segment
        revenue_by_segment = self.df.groupby('market_segment')['total_revenue'].mean().to_dict()
        insights['revenue_by_segment'] = revenue_by_segment
        
        # 2. Cancellation rate
        overall_cancellation = (self.df['is_canceled'].sum() / len(self.df)) * 100
        insights['overall_cancellation_rate'] = overall_cancellation
        
        # Cancellation by hotel type
        cancel_by_hotel = self.df.groupby('hotel')['is_canceled'].mean().to_dict()
        insights['cancellation_by_hotel'] = cancel_by_hotel
        
        # Cancellation by booking channel
        cancel_by_channel = self.df.groupby('market_segment')['is_canceled'].mean().to_dict()
        insights['cancellation_by_channel'] = cancel_by_channel
        
        # Cancellation by lead time
        cancel_by_lead = self.df.groupby('lead_time_bin')['is_canceled'].mean().to_dict()
        insights['cancellation_by_lead_time'] = cancel_by_lead
        
        # 3. Geographical distribution
        country_bookings = self.df.groupby('country')['hotel'].count().sort_values(ascending=False).head(10).to_dict()
        insights['top_countries'] = country_bookings
        
        # 4. Booking lead time distribution
        lead_time_stats = self.df['lead_time'].describe().to_dict()
        insights['lead_time_stats'] = lead_time_stats
        
        # 5. Additional insights
        # Length of stay
        length_stats = self.df['total_stays'].describe().to_dict()
        insights['length_of_stay_stats'] = length_stats
        
        # Booking channel performance
        segment_bookings = self.df.groupby('market_segment')['hotel'].count()
        segment_percentage = (segment_bookings / segment_bookings.sum() * 100).to_dict()
        insights['booking_channel_percentage'] = segment_percentage
        
        # ADR by customer type
        adr_by_customer = self.df.groupby('customer_type')['adr'].mean().to_dict()
        insights['adr_by_customer_type'] = adr_by_customer
        
        # Store insights
        self.insights = insights
        return insights
    
    def save_insights(self, output_path: str = "output/insights.json"):
        """Save insights to a JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.insights, f, indent=4)
        return output_path
    
    def generate_plots(self):
        """Generate various plots for the insights."""
        # Create a directory for plots
        plots_dir = os.path.join(OUTPUT_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Revenue trends over time
        plt.figure(figsize=(12, 6))
        self.df.groupby(pd.Grouper(key='arrival_date', freq='M'))['total_revenue'].sum().plot()
        plt.title('Monthly Revenue Trends')
        plt.xlabel('Month')
        plt.ylabel('Total Revenue')
        plt.savefig(os.path.join(plots_dir, 'monthly_revenue.png'))
        plt.close()
        
        # 2. Cancellation rate by hotel type
        plt.figure(figsize=(10, 6))
        self.df.groupby('hotel')['is_canceled'].mean().plot(kind='bar')
        plt.title('Cancellation Rate by Hotel Type')
        plt.xlabel('Hotel Type')
        plt.ylabel('Cancellation Rate')
        plt.savefig(os.path.join(plots_dir, 'cancellation_by_hotel.png'))
        plt.close()
        
        # 3. Top 10 countries
        plt.figure(figsize=(12, 6))
        top_countries = self.df.groupby('country')['hotel'].count().sort_values(ascending=False).head(10)
        top_countries.plot(kind='bar')
        plt.title('Top 10 Countries by Number of Bookings')
        plt.xlabel('Country')
        plt.ylabel('Number of Bookings')
        plt.savefig(os.path.join(plots_dir, 'top_countries.png'))
        plt.close()
        
        # 4. Lead time distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['lead_time'], bins=20)
        plt.title('Distribution of Booking Lead Time')
        plt.xlabel('Lead Time (days)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(plots_dir, 'lead_time_distribution.png'))
        plt.close()
        
        # 5. ADR by customer type
        plt.figure(figsize=(10, 6))
        self.df.groupby('customer_type')['adr'].mean().plot(kind='bar')
        plt.title('Average Daily Rate by Customer Type')
        plt.xlabel('Customer Type')
        plt.ylabel('Average Daily Rate')
        plt.savefig(os.path.join(plots_dir, 'adr_by_customer.png'))
        plt.close()
        
        return plots_dir


# Vector Database and RAG System
class BookingQASystem:
    def __init__(self, insights: Dict[str, Any], chroma_dir: str = CHROMA_DB_DIR):
        """Initialize the QA system with insights and a ChromaDB directory."""
        self.insights = insights
        self.chroma_dir = chroma_dir
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.llm_model = "google/flan-t5-base"  # A smaller model for easier deployment
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        
        # Create the vector store
        self.vector_store = None
        self.qa_chain = None
    
    def prepare_documents(self):
        """Prepare documents from insights for vector storage."""
        documents = []
        
        # Convert insights to documents
        for key, value in self.insights.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    text = f"{key} - {subkey}: {subvalue}"
                    documents.append(Document(page_content=text, metadata={"category": key, "subcategory": subkey}))
            else:
                text = f"{key}: {value}"
                documents.append(Document(page_content=text, metadata={"category": key}))
        
        # Add some general description documents
        documents.append(Document(
            page_content="This dataset contains booking information for hotels including city hotels and resort hotels. "
                       "It includes information such as booking dates, length of stay, number of guests, and more.",
            metadata={"category": "description"}
        ))
        
        return documents
    
    def create_vector_store(self, documents):
        """Create a vector store from the provided documents."""
        # Create a langchain Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.chroma_dir, "langchain_store")
        )
        
        # Persist the vector store
        self.vector_store.persist()
        return self.vector_store
    
    def setup_qa_chain(self):
        """Set up the QA chain with the vector store and LLM."""
        # Initialize the LLM
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
        
        # Create the pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )
        
        # Create the LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create the prompt template
        prompt_template = """
        You are a hotel booking data analyst. Use the provided context to answer the question.
        
        Context: {context}
        
        Question: {question}
        
        Answer: 
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> str:
        """Query the QA system with a question."""
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        # Get the answer
        result = self.qa_chain({"query": question})
        return result["result"]
    
    def calculate_response_time(self, question: str) -> Dict[str, Union[str, float]]:
        """Calculate the response time for a question."""
        start_time = time.time()
        answer = self.query(question)
        end_time = time.time()
        
        return {
            "question": question,
            "answer": answer,
            "response_time": end_time - start_time
        }
    
    def evaluate_system(self, test_questions: List[str]) -> Dict[str, Any]:
        """Evaluate the system with a set of test questions."""
        results = []
        total_time = 0
        
        for question in test_questions:
            result = self.calculate_response_time(question)
            results.append(result)
            total_time += result["response_time"]
        
        avg_time = total_time / len(test_questions) if test_questions else 0
        
        return {
            "results": results,
            "average_response_time": avg_time
        }


# API Implementation
app = FastAPI(title="Hotel Booking Analytics and QA API")

# Models for API
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    response_time: float

class AnalyticsResponse(BaseModel):
    insights: Dict[str, Any]

# Global variables for the system
processor = None
qa_system = None

@app.on_event("startup")
async def startup_event():
    global processor, qa_system
    
    # Initialize the processor
    processor = HotelBookingProcessor()
    processor.load_data()
    processor.clean_data()
    insights = processor.calculate_insights()
    processor.save_insights()
    processor.generate_plots()
    
    # Initialize the QA system
    qa_system = BookingQASystem(insights)
    documents = qa_system.prepare_documents()
    qa_system.create_vector_store(documents)
    qa_system.setup_qa_chain()

@app.post("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get analytics insights."""
    global processor
    
    if not processor:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    return {"insights": processor.insights}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the hotel booking data."""
    global qa_system
    
    if not qa_system:
        raise HTTPException(status_code=500, detail="QA system not initialized")
    
    result = qa_system.calculate_response_time(request.question)
    
    return {
        "answer": result["answer"],
        "response_time": result["response_time"]
    }

@app.get("/health")
async def health_check():
    """Check the health of the system."""
    global processor, qa_system
    
    status = {
        "processor": processor is not None,
        "qa_system": qa_system is not None,
        "vector_store": qa_system.vector_store is not None if qa_system else False,
        "qa_chain": qa_system.qa_chain is not None if qa_system else False
    }
    
    all_healthy = all(status.values())
    
    if all_healthy:
        return {"status": "healthy", "details": status}
    else:
        return {"status": "unhealthy", "details": status}

# Main function to run the system
def main():
    """Main function to run the system."""
    # Initialize the processor
    processor = HotelBookingProcessor()
    processor.load_data()
    processor.clean_data()
    insights = processor.calculate_insights()
    processor.save_insights()
    processor.generate_plots()
    
    # Initialize the QA system
    qa_system = BookingQASystem(insights)
    documents = qa_system.prepare_documents()
    qa_system.create_vector_store(documents)
    qa_system.setup_qa_chain()
    
    # Test the system
    test_questions = [
        "Show me total revenue for July 2017.",
        "Which locations had the highest booking cancellations?",
        "What is the average price of a hotel booking?",
        "What is the cancellation rate for city hotels?",
        "Which market segment has the highest booking percentage?"
    ]
    
    evaluation = qa_system.evaluate_system(test_questions)
    
    print(f"Average response time: {evaluation['average_response_time']:.2f} seconds")
    for i, result in enumerate(evaluation['results']):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Response time: {result['response_time']:.2f} seconds")
    
    # Start the API
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()