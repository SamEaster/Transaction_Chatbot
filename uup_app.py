from flask import Flask, request, jsonify
import os
from uup_config import Config
from uup_agent import MongoAgent

app = Flask(__name__)
app.config.from_object(Config)

mongo_agent = MongoAgent()

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({'message': 'NLP to MongoDB API is running!', 'status': 'success'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'nlp-to-mongodb'})

@app.route('/query', methods=['POST'])
def process_query():
    """Process natural language query with user item filter"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data or 'question' not in data or 'item' not in data:
        return jsonify({'error': 'Missing "question" or "item" in request'}), 400
    
    question = data['question'].strip()
    item = data['item'].strip()
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    if not item:
        return jsonify({'error': 'Item cannot be empty'}), 400
    
    try:
        response = mongo_agent.process_query(question, item)
        return jsonify({
            'question': question,
            'item': item,
            'response_after': response,
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Query processing error: {str(e)}'}), 500

@app.route('/user_info', methods=['POST'])
def get_user_info():
    """Get user information from source database"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if not data or 'item' not in data:
        return jsonify({'error': 'Missing "item" in request'}), 400
    
    item = data['item'].strip()
    
    if not item:
        return jsonify({'error': 'Item cannot be empty'}), 400
    
    try:
        user_info = mongo_agent.get_user_info(item)
        
        if user_info:
            return jsonify({
                'user_info': user_info,
                'status': 'success'
            }), 200
        else:
            return jsonify({
                'error': 'No user found with the provided item'
            }), 404
    
    except Exception as e:
        return jsonify({'error': f'Error retrieving user info: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        mongo_agent.close_connection()