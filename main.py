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
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        
        # The vector store and QA chain will be created later
        self.vector_store = None
        self.qa_chain = None
    
    def prepare_documents(self):
        """Prepare documents from insights and stored graphs for vector storage."""
        documents = []
        
        # Convert insights to text documents
        for key, value in self.insights.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    # Format numerical values for better readability
                    if isinstance(subvalue, (int, float)):
                        if key == 'revenue_by_hotel' or key == 'revenue_by_room' or key == 'revenue_by_segment':
                            formatted_value = f"${subvalue:.2f}"
                        elif 'rate' in key or 'percentage' in key:
                            formatted_value = f"{subvalue:.2f}%"
                        else:
                            formatted_value = f"{subvalue:.2f}"
                    else:
                        formatted_value = str(subvalue)
                    
                    text = f"{key} - {subkey}: {formatted_value}"
                    documents.append(Document(page_content=text, metadata={"category": key, "subkey": subkey}))
            else:
                # Format numerical values
                if isinstance(value, (int, float)):
                    if 'revenue' in key:
                        formatted_value = f"${value:.2f}"
                    elif 'rate' in key or 'percentage' in key:
                        formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                text = f"{key}: {formatted_value}"
                documents.append(Document(page_content=text, metadata={"category": key}))
        
        # Add more detailed descriptions for better context
        documents.append(Document(
            page_content="This dataset contains booking information for hotels, including city hotels and resort hotels. "
                         "It includes details such as booking dates, length of stay, guest numbers, revenue, and more.",
            metadata={"category": "description"}
        ))
        
        # Better descriptions for key metrics
        documents.append(Document(
            page_content="Overall cancellation rate represents the percentage of bookings that were canceled across all hotels.",
            metadata={"category": "cancellation_rate_description"}
        ))
        
        documents.append(Document(
            page_content="Revenue by market segment shows the average revenue generated by each market segment such as Online TA, Offline TA/TO, Direct, Corporate, Groups, etc.",
            metadata={"category": "market_segment_description"}
        ))
        
        # Add graph documents from the plots folder with detailed descriptions
        if os.path.exists(PLOTS_DIR):
            for file in os.listdir(PLOTS_DIR):
                if file.endswith('.png'):
                    file_path = os.path.join(PLOTS_DIR, file)
                    with open(file_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    # Create descriptive text based on the graph name
                    graph_name = file.split('.')[0]
                    if graph_name == 'monthly_revenue':
                        description = "Monthly revenue trends showing total revenue earned by hotels over time."
                    elif graph_name == 'cancellation_by_hotel':
                        description = "Cancellation rates by hotel type (city hotels vs. resort hotels)."
                    elif graph_name == 'top_countries':
                        description = "Top 10 countries by number of bookings."
                    elif graph_name == 'lead_time_distribution':
                        description = "Distribution of booking lead times showing how far in advance guests make reservations."
                    elif graph_name == 'adr_by_customer':
                        description = "Average daily rate by customer type."
                    else:
                        description = f"Graph showing data for {graph_name.replace('_', ' ')}."
                    
                    text = f"Graph: {graph_name} - {description}"
                    # Store the image and description as metadata for better retrieval
                    documents.append(Document(
                        page_content=text, 
                        metadata={
                            "type": "graph", 
                            "image_base64": encoded_string,
                            "graph_name": graph_name,
                            "description": description
                        }
                    ))
        return documents
    
    def create_vector_store(self, documents):
        """Create a vector store from the provided documents."""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.chroma_dir, "langchain_store")
        )
        self.vector_store.persist()
        return self.vector_store
    
    def setup_qa_chain(self):
        """Set up the QA chain with the vector store and LLM."""
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
        
        # Updated prompt template for human-friendly and complete answers
        prompt_template = """
You are a hotel booking data analyst. Provide a clear, descriptive, and human-friendly answer to the following question.
Be specific and precise with your response. Format numbers appropriately (e.g., use dollar signs for monetary amounts, percentage signs for rates).

If you're referring to a specific market segment, hotel type, or room type, always mention it by its exact name.
For example, instead of saying "a market segment," state "the Online TA market segment" if that's what the data shows.

If asked about trends, rates, or statistics, provide the exact figures when available, formatted properly:
- For currency values: $X,XXX.XX
- For percentages: XX.XX%
- For counts: X,XXX

If the necessary data is missing, say "I currently don't have this data available. I will update you once it's available."

Use complete sentences and, if applicable, mention that there is a graph available to illustrate the answer.

Context: {context}
Question: {question}
Answer:
"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain using the vector store as the retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),  # Increase k for better context
            chain_type_kwargs={"prompt": PROMPT}
        )
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the QA system with a question and attempt to attach a relevant graph if available."""
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain() first.")
        
        # First, check if we have relevant documents for this query
        retrieved_docs = self.vector_store.similarity_search(question, k=5)
        
        # Check if we have any information at all
        has_relevant_info = False
        for doc in retrieved_docs:
            if doc.metadata.get("type") != "graph":  # If it's not just a graph
                has_relevant_info = True
                break
        
        # Get the textual answer from the QA chain
        result = self.qa_chain({"query": question})
        answer_text = result.get("result", "").strip()
        
        # Handle missing or empty answer
        if not answer_text or not has_relevant_info:
            answer_text = "I currently don't have this data available. I will update you once it's available."
        
        # Post-process the answer to ensure formatting meets requirements
        answer_text = self._format_answer(answer_text)
        
        # Attempt to find a related graph from the vector store
        graph_image = None
        graph_description = None
        
        for doc in retrieved_docs:
            if doc.metadata.get("type") == "graph":
                graph_image = doc.metadata.get("image_base64")
                graph_description = doc.metadata.get("description", "")
                break
        
        # If we have a graph, add a reference to it in the answer
        if graph_image and not "I currently don't have this data available" in answer_text:
            if not answer_text.endswith("."):
                answer_text += "."
            answer_text += f" A graph is available to illustrate this data."
        
        return {
            "answer": answer_text, 
            "graph": graph_image, 
            "graph_description": graph_description
        }
    
    def _format_answer(self, answer: str) -> str:
        """Format the answer to ensure consistent styling and proper handling of numbers."""
        # Ensure proper formatting for percentages
        import re
        
        # Format percentages
        percentage_pattern = r'(\d+(\.\d+)?)\s*%'
        answer = re.sub(percentage_pattern, r'**\1%**', answer)
        
        # Format dollar amounts
        dollar_pattern = r'\$\s*(\d+(\.\d+)?)'
        answer = re.sub(dollar_pattern, r'**$\1**', answer)
        
        # Look for market segment names and bold them
        segments = ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Groups", "Aviation", "Complementary"]
        for segment in segments:
            if segment in answer:
                answer = answer.replace(segment, f"**{segment}**")
        
        # Look for hotel types and bold them
        hotel_types = ["City Hotel", "Resort Hotel"]
        for hotel_type in hotel_types:
            if hotel_type in answer:
                answer = answer.replace(hotel_type, f"**{hotel_type}**")
        
        return answer
    
    def calculate_response_time(self, question: str) -> Dict[str, Union[str, float, Any]]:
        """Calculate the response time for a question."""
        start_time = time.time()
        result = self.query(question)
        end_time = time.time()
        
        result["response_time"] = end_time - start_time
        return result
    
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

# Updated API models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    response_time: float
    graph: Union[str, None] = None  # Base64-encoded graph image if available
    graph_description: Union[str, None] = None  # Description of the graph

class AnalyticsResponse(BaseModel):
    insights: Dict[str, Any]


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the hotel booking data."""
    global qa_system
    if not qa_system:
        raise HTTPException(status_code=500, detail="QA system not initialized")
    
    result = qa_system.calculate_response_time(request.question)
    

    formatted_answer = result["answer"]
    if not formatted_answer.endswith(f"(Response time: {result['response_time']:.2f} seconds)"):
        formatted_answer += f"\n\n(Response time: {result['response_time']:.2f} seconds)"
    
    return {
        "answer": formatted_answer,
        "response_time": result["response_time"],
        "graph": result.get("graph"),
        "graph_description": result.get("graph_description")
    }

@app.get("/qa-interface")
async def qa_interface():
    """Serve a simple UI for interacting with the QA system."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hotel Booking Analytics QA</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .container {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 20px;
            }
            .question-form {
                margin-bottom: 20px;
            }
            input[type="text"] {
                width: 80%;
                padding: 10px;
                font-size: 16px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                font-size: 16px;
            }
            .answer {
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .graph-container {
                margin-top: 20px;
                text-align: center;
            }
            .graph-container img {
                max-width: 100%;
                border: 1px solid #ddd;
            }
            .response-time {
                font-size: 12px;
                color: #666;
                margin-top: 10px;
            }
            .bold {
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hotel Booking Analytics QA</h1>
            <div class="question-form">
                <input type="text" id="question" placeholder="Ask a question about hotel bookings...">
                <button onclick="askQuestion()">Ask</button>
            </div>
            <div id="answer-container" style="display: none;">
                <div class="answer" id="answer-text"></div>
                <div class="graph-container" id="graph-container" style="display: none;">
                    <h3>Related Graph</h3>
                    <img id="graph-image">
                    <p id="graph-description"></p>
                </div>
                <div class="response-time" id="response-time"></div>
            </div>
        </div>
        
        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                if (!question) return;
                
                document.getElementById('answer-text').innerHTML = 'Thinking...';
                document.getElementById('answer-container').style.display = 'block';
                document.getElementById('graph-container').style.display = 'none';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question }),
                    });
                    
                    const data = await response.json();
                    
                    // Display the answer
                    document.getElementById('answer-text').innerHTML = data.answer.replace(/\*\*(.*?)\*\*/g, '<span class="bold">$1</span>');
                    
                    // Display the graph if available
                    if (data.graph) {
                        document.getElementById('graph-container').style.display = 'block';
                        document.getElementById('graph-image').src = 'data:image/png;base64,' + data.graph;
                        
                        if (data.graph_description) {
                            document.getElementById('graph-description').textContent = data.graph_description;
                        } else {
                            document.getElementById('graph-description').textContent = '';
                        }
                    }
                    
                    // Display the response time
                    document.getElementById('response-time').textContent = `Response time: ${data.response_time.toFixed(2)} seconds`;
                } catch (error) {
                    document.getElementById('answer-text').innerHTML = 'Error: ' + error.message;
                }
            }
            
            // Allow pressing Enter key to ask question
            document.getElementById('question').addEventListener('keypress', function(event) {
                if (event.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
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


from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean them up on shutdown."""
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
    
    yield
    

    pass

app = FastAPI(
    title="Hotel Booking Analytics and QA API",
    description="An API for querying hotel booking data analytics with natural language",
    version="2.0",
    lifespan=lifespan
)

def main_notebook():
    """Main function to run the system without starting the web server."""
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

    # Test the system with improved sample questions
    test_questions = [
        "Show me total revenue for July 2017.",
        "Which locations had the highest booking cancellations?",
        "What is the average price of a hotel booking?",
        "What is the cancellation rate for city hotels?",
        "Which market segment has the highest booking percentage?",
        "Which market segment has the highest cancellation rate?",
        "What are the trends in monthly revenue?",
        "How does the lead time affect cancellation rates?",
        "Which customer type generates the highest revenue?",
        "What's the average length of stay at resort hotels?"
    ]

    # Run evaluation
    evaluation = qa_system.evaluate_system(test_questions)
    print(f"Average response time: {evaluation['average_response_time']:.2f} seconds")
    print(f"\n{'='*50}\n")
    print(f"EVALUATION RESULTS:\n")
    
    for i, result in enumerate(evaluation['results']):
        print(f"\n{'='*50}")
        print(f"Question {i+1}: {test_questions[i]}")
        print(f"Answer: {result['answer']}")
        print(f"Has Graph: {'Yes' if result.get('graph') else 'No'}")
        print(f"Response time: {result['response_time']:.2f} seconds")
    
    # Return the processor and qa_system
    return processor, qa_system

# Run the main notebook function if script is executed directly
if __name__ == "__main__":
    processor, qa_system = main_notebook()