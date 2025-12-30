# school_ai_model.py (UPDATED with frontend support)
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re
from flask import Flask, request, jsonify, render_template
import subprocess
import os

class SchoolAIModel:
    def __init__(self, db_path="./chroma_school_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.client.get_collection("school_management")
        
        # Load original data for detailed queries
        self.original_data = {}
        if os.path.exists("data_cache"):
            for file in ["schools", "classes", "students", "teachers", "parents"]:
                try:
                    self.original_data[file] = pd.read_json(f"data_cache/{file}.json")
                except:
                    pass
    
    def query_vector_db(self, query_text, n_results=10, filter_type=None):
        """Query the vector database"""
        query_embedding = self.embedder.encode(query_text).tolist()
        
        where_filter = {"type": filter_type} if filter_type else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        return results
    
    def extract_query_type(self, query):
        """Extract query type from natural language"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["student", "pupil", "learner", "enrolled", "admission"]):
            return "student"
        elif any(word in query_lower for word in ["teacher", "faculty", "instructor", "educator"]):
            return "teacher"
        elif any(word in query_lower for word in ["parent", "guardian", "mother", "father"]):
            return "parent"
        elif any(word in query_lower for word in ["class", "grade", "section", "batch", "room"]):
            return "class"
        elif any(word in query_lower for word in ["school", "institute", "academy", "college"]):
            return "school"
        
        return None
    
    def extract_specific_info(self, query):
        """Extract specific information from query"""
        patterns = {
            "school_code": r"(SCH\d{3})",
            "student_code": r"(STU\d{3})",
            "teacher_code": r"(TCH\d{3})",
            "parent_code": r"(PAR\d{3})",
            "class_name": r"(Grade\s+\d+|Class\s+\d+)",
            "city": r"(Pune|Mumbai|Kochi|Delhi|Chennai|Trivandrum|Gurugram|Jaipur|Patna|Bengaluru)",
            "state": r"(Maharashtra|Kerala|Delhi|Tamil Nadu|Haryana|Rajasthan|Bihar|Karnataka)"
        }
        
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                extracted[key] = match.group(1)
        
        return extracted
    
    def execute_sql_like_query(self, query_type, extracted_info):
        """Execute SQL-like queries on original data"""
        results = []
        
        if query_type == "student" and "student_code" in extracted_info:
            if "students" in self.original_data:
                df = self.original_data["students"]
                student_data = df[df['student_code'] == extracted_info["student_code"]]
                if not student_data.empty:
                    results.append({
                        "type": "student_detail",
                        "data": student_data.iloc[0].to_dict()
                    })
        
        elif query_type == "teacher" and "teacher_code" in extracted_info:
            if "teachers" in self.original_data:
                df = self.original_data["teachers"]
                teacher_data = df[df['teacher_code'] == extracted_info["teacher_code"]]
                if not teacher_data.empty:
                    results.append({
                        "type": "teacher_detail",
                        "data": teacher_data.iloc[0].to_dict()
                    })
        
        return results
    
    def get_statistics(self, query_type=None):
        """Get statistics about the data"""
        stats = {}
        
        if "students" in self.original_data:
            df = self.original_data["students"]
            stats["total_students"] = len(df)
            stats["students_by_gender"] = df['gender'].value_counts().to_dict()
            
            # Count by school
            if 'school_code' in df.columns:
                stats["students_by_school"] = df['school_code'].value_counts().to_dict()
        
        if "teachers" in self.original_data:
            df = self.original_data["teachers"]
            stats["total_teachers"] = len(df)
            stats["teachers_by_subject"] = df['subject'].value_counts().to_dict()
        
        if "schools" in self.original_data:
            df = self.original_data["schools"]
            stats["total_schools"] = len(df)
            stats["schools_by_state"] = df['school_state'].value_counts().to_dict()
        
        return stats
    
    def process_query(self, query_text):
        """Process a natural language query"""
        # Step 1: Extract query type
        query_type = self.extract_query_type(query_text)
        
        # Step 2: Extract specific information
        extracted_info = self.extract_specific_info(query_text)
        
        # Step 3: Query vector database
        vector_results = self.query_vector_db(query_text, n_results=20, filter_type=query_type)
        
        # Step 4: Execute specific queries if codes are found
        specific_results = []
        if any(key in extracted_info for key in ["student_code", "teacher_code", "parent_code"]):
            specific_results = self.execute_sql_like_query(query_type, extracted_info)
        
        # Step 5: Check for statistical queries
        stats_results = []
        if any(word in query_text.lower() for word in ["total", "count", "number", "statistics", "how many"]):
            stats_results = self.get_statistics(query_type)
        
        # Step 6: Format results
        response = {
            "query": query_text,
            "query_type": query_type,
            "extracted_info": extracted_info,
            "vector_results": self.format_vector_results(vector_results),
            "specific_results": specific_results,
            "statistics": stats_results if stats_results else None
        }
        
        return response
    
    def format_vector_results(self, vector_results):
        """Format vector results for display"""
        if not vector_results or not vector_results['documents']:
            return []
        
        formatted = []
        for i in range(len(vector_results['documents'][0])):
            formatted.append({
                "content": vector_results['documents'][0][i],
                "metadata": vector_results['metadatas'][0][i],
                "distance": vector_results['distances'][0][i]
            })
        
        return formatted

# Flask API
app = Flask(__name__, template_folder='templates')
ai_model = SchoolAIModel()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle natural language queries"""
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Process the query
        response = ai_model.process_query(query_text)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get overall statistics"""
    stats = ai_model.get_statistics()
    return jsonify(stats)

@app.route('/search', methods=['POST'])
def search():
    """Search with filters"""
    data = request.json
    query = data.get('query', '')
    entity_type = data.get('type', None)
    limit = data.get('limit', 10)
    
    results = ai_model.query_vector_db(query, n_results=limit, filter_type=entity_type)
    formatted = ai_model.format_vector_results(results)
    
    return jsonify({"results": formatted})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "database": "connected"})

if __name__ == '__main__':
    print("Starting School Management AI Model...")
    print("Frontend available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)