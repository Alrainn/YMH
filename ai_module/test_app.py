import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test that the index route returns a 200 OK status."""
    response = client.get('/')
    assert response.status_code == 200

def test_chat_endpoint_missing_message(client):
    """Test the chat endpoint when the message field is missing."""
    response = client.post('/api/chat', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_chat_endpoint_valid_message(client):
    """Test the chat endpoint with a valid message."""
    response = client.post('/api/chat', json={
        'message': 'Merhaba',
        'session_id': 'test_session',
        'conversation_history': []
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data

def test_drug_info_endpoint_missing_fields(client):
    """Test the drug info endpoint with missing required fields."""
    response = client.post('/api/drug-info', json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_drug_info_endpoint_valid_request(client):
    """Test the drug info endpoint with a valid request."""
    response = client.post('/api/drug-info', json={
        'drug_name_en': 'aspirin',
        'question_tr': 'Yan etkileri nelerdir?',
        'session_id': 'test_session',
        'conversation_history': []
    })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'response' in data 