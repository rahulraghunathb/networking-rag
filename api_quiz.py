"""
Quiz Mode API - Handles quiz generation and answer checking with web citations
"""
from __future__ import annotations

from pathlib import Path
import random
import uuid
import os
from typing import List, Optional, Dict
from urllib.parse import quote_plus

import aiohttp
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ollama import Client
import json
import re


# Constants
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
OLLAMA_MODEL = None
QUIZ_TOP_K = 12
MIN_QUESTIONS = 1
MAX_QUESTIONS = 10
WEB_SEARCH_TIMEOUT = 10

# Hardcoded topics from the networking vector database
HARDCODED_TOPICS = [
    "Firewalls",
    "DNS (Domain Name System)",
    "TCP/IP Protocol",
    "Network Security",
    "Encryption and SSL/TLS",
    "VPN (Virtual Private Network)",
    "DDoS Attacks",
    "HTTP and HTTPS",
    "Network Routing",
    "OSI Model",
    "IP Addressing and Subnetting",
    "Network Authentication",
    "Wireless Security (WPA/WPA2)",
    "Intrusion Detection Systems (IDS)",
    "Load Balancing",
    "Network Protocols",
    "Packet Switching",
    "Network Topology",
    "Cybersecurity Threats",
    "Email Security (SMTP, POP3, IMAP)"
]


# Pydantic Models
class ContextItem(BaseModel):
    rank: int
    source: Optional[str] = None
    page: Optional[int] = None
    text: str


class WebCitation(BaseModel):
    title: str
    url: str
    snippet: str
    source: str


class QuizRequest(BaseModel):
    topic: Optional[str] = None
    question_type: str = 'multiple_choice'
    count: int = 1


class QuizQuestion(BaseModel):
    id: str
    type: str
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    citations: List[dict]


class QuizResponse(BaseModel):
    total_questions: int
    questions: List[QuizQuestion]


class QuizCheckRequest(BaseModel):
    question_id: str
    user_answer: str


class QuizCheckResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    user_grade: str
    feedback: str
    citations: List[dict]
    web_citations: List[dict]
    confidence_score: float


# Global instances
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None
_ollama_client: Optional[Client] = None
_quiz_cache: Dict[str, Dict[str, any]] = {}


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model


def _get_ollama_model() -> str:
    global OLLAMA_MODEL
    if OLLAMA_MODEL is None:
        load_dotenv()
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    return OLLAMA_MODEL


def _check_ollama_health() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        client = _load_ollama_client()
        model_name = _get_ollama_model()
        client.show(model_name)
        return True
    except Exception as e:
        global _ollama_client
        _ollama_client = None
        return False


def _load_ollama_client() -> Client:
    global _ollama_client
    if _ollama_client is None:
        load_dotenv()
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        _ollama_client = Client(host=host)
    return _ollama_client


def _load_collection() -> chromadb.Collection:
    """Load the chroma collection for context retrieval."""
    global _collection
    if _collection is None:
        print(f"Loading chroma collection: {COLLECTION_NAME} from {PERSIST_DIR}")
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
        _collection = client.get_collection(COLLECTION_NAME)
        print("ChromaDB collection loaded successfully")
    return _collection


def _embed_query(text: str) -> List[float]:
    print(f"Generating embedding for text: {text[:100]}...")
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=False, normalize_embeddings=True)
    embedding = vec[0].tolist()
    print(f"Generated embedding of length: {len(embedding)}")
    return embedding


def _fetch_context(query_embedding: List[float], top_k: int) -> List[ContextItem]:
    print(f"Fetching context from vector database, top_k={top_k}")
    collection = _load_collection()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        print(f"Retrieved {len(documents)} documents from vector database")
        
        items: List[ContextItem] = []
        for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source = meta.get("source") if isinstance(meta, dict) else None
            page = meta.get("page") if isinstance(meta, dict) else None
            text = (doc or "").strip()
            items.append(ContextItem(rank=idx, source=source, page=page, text=text))
            print(f"Context {idx}: source={source}, page={page}, text_length={len(text)}")
        
        return items
    except Exception as e:
        print(f"Error fetching context from vector database: {e}")
        raise


async def _search_web(query: str, max_results: int = 3) -> List[WebCitation]:
    """Search the web for relevant information and return citations."""
    try:
        search_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"

        timeout = aiohttp.ClientTimeout(total=WEB_SEARCH_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    citations = []

                    if data.get('AbstractText') and data['AbstractText'].strip():
                        citations.append(WebCitation(
                            title=data.get('Answer', query)[:100],
                            url=data.get('AbstractURL', ''),
                            snippet=data['AbstractText'][:200],
                            source='DuckDuckGo'
                        ))

                    if data.get('RelatedTopics'):
                        for topic in data['RelatedTopics'][:max_results-1]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                citations.append(WebCitation(
                                    title=topic['Text'][:100],
                                    url=topic.get('FirstURL', ''),
                                    snippet=topic['Text'][:200],
                                    source='DuckDuckGo'
                                ))

                    return citations[:max_results]
                else:
                    return []
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def _llm_generate_question(context_text: str, question_type: str, topic: str) -> Dict[str, any]:
    """Use Ollama to generate intelligent quiz questions based on context."""
    print(f"Generating question using LLM for topic: {topic}, type: {question_type}")
    if not _check_ollama_health():
        raise HTTPException(status_code=503, detail="Ollama service is not available. Please ensure Ollama is running and the model is loaded.")
    
    client = _load_ollama_client()
    model_name = _get_ollama_model()
    
    prompt = _get_prompt_for_type(question_type, topic, context_text)
    
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 500}
        )
        print("LLM response received")
        
        result_text = response["message"]["content"].strip()
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return _fallback_question_generation(context_text, question_type, topic)
            
    except Exception as e:
        print(f"LLM question generation failed: {e}")
        return _fallback_question_generation(context_text, question_type, topic)


def _fallback_question_generation(context_text: str, question_type: str, topic: str) -> Dict[str, any]:
    """Fallback question generation using the original hardcoded logic."""
    # Extract a key sentence from context
    sentences = [s.strip() for s in context_text.split('.') if s.strip() and len(s.strip()) > 20]
    key_sentence = sentences[0] if sentences else f"{topic} is important in networking."
    
    if question_type == 'multiple_choice':
        # Create simple distractors without the removed function
        distractors = [
            f"{topic} is not important in networking",
            f"{topic} only works with specific hardware",
            f"{topic} is unrelated to modern networks"
        ]
        return {
            "question": f"What is the primary function of {topic}?",
            "correct_answer": key_sentence[:100],
            "options": [key_sentence[:100]] + distractors,
            "explanation": f"Based on the context about {topic}"
        }
    elif question_type == 'true_false':
        return {
            "question": f"True or False: {key_sentence[:80]}",
            "options": ["True", "False"],
            "correct_answer": "True",
            "explanation": f"This statement is derived from the context about {topic}"
        }
    else:  # open_ended
        return {
            "question": f"Explain the importance of {topic} in networking.",
            "correct_answer": key_sentence,
            "explanation": f"Key information from the context about {topic}"
        }

def _generate_question_from_context(contexts: List[ContextItem], question_type: str, topic: Optional[str] = None) -> QuizQuestion:
    """Generate a quiz question using LLM or raise error if unavailable."""
    print(f"Generating question from context - type: {question_type}, topic: {topic}")
    
    if not contexts:
        print("No contexts provided, using fallback context")
        contexts = [ContextItem(rank=1, source=None, page=None, text=f"Key facts about {topic or 'networking'}.")]

    # Combine all context texts for LLM
    context_text = "\n\n".join([f"[{c.rank}] {c.text}" for c in contexts if c.text])
    
    citations = [
        {"source": c.source or "context", "page": c.page, "rank": c.rank}
        for c in contexts[:3] if c.text and c.text.strip()
    ]

    question_id = str(uuid.uuid4())
    topic_label = (topic or "networking").title()

    try:
        # Use LLM-powered question generation
        llm_result = _llm_generate_question(context_text, question_type, topic_label)
        print("LLM question generation successful")
        
        # Convert LLM result to QuizQuestion format
        if question_type == 'multiple_choice':
            question = QuizQuestion(
                id=question_id,
                type='multiple_choice',
                question=llm_result.get('question', f"What is the primary function of {topic_label}?"),
                options=llm_result.get('options', [llm_result.get('correct_answer', 'Answer'), 'Wrong 1', 'Wrong 2', 'Wrong 3']),
                correct_answer=llm_result.get('correct_answer', 'Answer'),
                explanation=llm_result.get('explanation', f"Based on the context about {topic_label}"),
                citations=citations
            )
        elif question_type == 'true_false':
            question = QuizQuestion(
                id=question_id,
                type='true_false',
                question=llm_result.get('question', f"True or False: {topic_label} is important."),
                options=["True", "False"],
                correct_answer=llm_result.get('correct_answer', 'True'),
                explanation=llm_result.get('explanation', f"This relates to {topic_label}"),
                citations=citations
            )
        else:  # open_ended
            question = QuizQuestion(
                id=question_id,
                type='open_ended',
                question=llm_result.get('question', f"Explain {topic_label}."),
                options=None,
                correct_answer=llm_result.get('correct_answer', f"{topic_label} is important in networking."),
                explanation=llm_result.get('explanation', f"Key information about {topic_label}"),
                citations=citations
            )
        
        print(f"Successfully generated question: {question.id}")
        return question
        
    except Exception as e:
        print(f"Error generating question: {e}")
        raise


def _llm_grade_open_ended_answer(question: str, user_answer: str, correct_answer: str, context: str) -> tuple[bool, str, float]:
    """Use Ollama to intelligently grade open-ended answers."""
    if not _check_ollama_health():
        return False, 'F', 0.0
    
    client = _load_ollama_client()
    model_name = _get_ollama_model()
    
    prompt = f"""Grade this student's answer for an open-ended networking question.

Question: {question}

Student's Answer: {user_answer}

Expected Answer: {correct_answer}

Context: {context[:500]}

Grade the answer on a scale of A-F and provide detailed feedback. Consider:
- Accuracy and correctness of technical details
- Depth of understanding shown
- Completeness of explanation
- Use of relevant networking terminology
- Structure and clarity of response

Respond in this exact JSON format:
{{
    "grade": "A",
    "is_correct": true,
    "confidence": 0.95,
    "feedback": "Detailed feedback explaining the grade and what was good/missing"
}}

Provide only ONE grade (A, B, C, D, or F), not multiple options. Be specific in your feedback."""

    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 400}
        )
        
        result_text = response["message"]["content"].strip()
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            grade = result.get('grade', 'F')
            is_correct = result.get('is_correct', False)
            confidence = result.get('confidence', 0.0)
            
            # Ensure grade is a single letter
            if grade not in ['A', 'B', 'C', 'D', 'F']:
                grade = 'C'  # Default fallback
                
            return is_correct, grade, confidence
        else:
            print("No valid JSON in LLM grading response")
            return False, 'C', 0.5
        
    except Exception as e:
        print(f"LLM answer grading failed: {e}")
        return False, 'C', 0.0


def _grade_answer(user_answer: str, correct_answer: str, question_type: str, question: str = "", context: str = "") -> tuple[bool, str, float]:
    """Grade answers using LLM for open-ended questions, exact match for others."""
    if not user_answer or not user_answer.strip():
        return False, 'F', 0.0

    if question_type in ['multiple_choice', 'true_false']:
        # Exact match for objective questions
        if user_answer.lower().strip() == correct_answer.lower().strip():
            return True, 'A', 1.0
        else:
            return False, 'F', 0.0
    else:  # open_ended
        # Use LLM-powered grading
        return _llm_grade_open_ended_answer(question, user_answer, correct_answer, context)


def _get_prompt_for_type(question_type: str, topic: str, context_text: str) -> str:
    """Get the appropriate prompt template based on question type."""
    if question_type == 'multiple_choice':
        return _get_multiple_choice_prompt(topic, context_text)
    elif question_type == 'true_false':
        return _get_true_false_prompt(topic, context_text)
    elif question_type == 'open_ended':
        return _get_open_ended_prompt(topic, context_text)
    else:
        return _get_multiple_choice_prompt(topic, context_text)

def _get_multiple_choice_prompt(topic: str, context_text: str) -> str:
    """Generate prompt for multiple choice questions."""
    return f"""Based on the following networking context about {topic}, generate a multiple choice question with 4 options (one correct, three plausible distractors).

Context: {context_text[:1000]}

Generate a question in this exact JSON format:
{{
    "question": "The question text here",
    "correct_answer": "The correct answer text",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "explanation": "Brief explanation of why the answer is correct"
}}

Make the distractors realistic and educational."""

def _get_true_false_prompt(topic: str, context_text: str) -> str:
    """Generate prompt for true/false questions."""
    return f"""Based on the following networking context about {topic}, generate a true/false question.

Context: {context_text[:1000]}

Generate a question in this exact JSON format:
{{
    "question": "True or False: [statement that can be clearly true or false]",
    "options": ["True", "False"],
    "correct_answer": "True",
    "explanation": "Brief explanation supporting the answer"
}}

The statement should be definitively true or false based on the context."""

def _get_open_ended_prompt(topic: str, context_text: str) -> str:
    """Generate prompt for open-ended questions."""
    return f"""Based on the following networking context about {topic}, generate an open-ended question that requires explanation.

Context: {context_text[:1000]}

Generate a question in this exact JSON format:
{{
    "question": "Explain/Describe [topic-related question requiring detailed answer]",
    "correct_answer": "A comprehensive answer based on the context",
    "explanation": "Key points that should be covered in the answer"
}}

The question should encourage detailed, thoughtful responses."""

async def _get_web_citations_for_feedback(question: str, topic: str) -> List[WebCitation]:
    """Get web citations for quiz feedback."""
    search_query = f"{topic} {question}" if topic else question
    return await _search_web(search_query, max_results=2)


def generate_quiz(topic: Optional[str], question_type: str, count: int) -> QuizResponse:
    """Generate quiz questions - randomly or topic-specific."""
    print(f"=== STARTING QUIZ GENERATION ===")
    print(f"Parameters: topic={topic}, question_type={question_type}, count={count}")
    
    try:
        # If no topic provided, randomly select from hardcoded topics
        if not topic or not topic.strip():
            topic_query = random.choice(HARDCODED_TOPICS)
            print(f"No topic provided, selected random topic: {topic_query}")
        else:
            topic_query = topic.strip()
        query_embedding = _embed_query(topic_query)
        contexts = _fetch_context(query_embedding, QUIZ_TOP_K)

        if not contexts:
            print("No context found for quiz generation")
            raise HTTPException(status_code=404, detail="No context for quiz generation")

        print(f"Found {len(contexts)} context items")

        questions = []
        num_questions = max(MIN_QUESTIONS, min(count, MAX_QUESTIONS))
        
        for i in range(num_questions):
            print(f"=== Generating question {i+1}/{num_questions} ===")
            
            # Randomize question type if 'random' is selected
            q_type = question_type if question_type != 'random' else random.choice(['multiple_choice', 'true_false', 'open_ended'])
            
            # Generate question with randomization
            q = _generate_question_from_context(contexts, q_type, topic_query)
            
            # Create context text for caching (same as used in question generation)
            context_text = "\n\n".join([f"[{c.rank}] {c.text}" for c in contexts if c.text])
            
            # Cache question data for checking later
            _quiz_cache[q.id] = {
                "correct_answer": q.correct_answer,
                "question_type": q.type,
                "explanation": q.explanation,
                "citations": q.citations,
                "topic": topic_query,
                "question": q.question,  # Store the question text
                "context": context_text  # Store the context for grading
            }
            questions.append(q)

        print(f"=== QUIZ GENERATION COMPLETE ===")
        print(f"Generated {len(questions)} questions successfully")
        return QuizResponse(total_questions=len(questions), questions=questions)
        
    except HTTPException:
        print(f"HTTP Exception in quiz generation", exc_info=True)
        raise
    except Exception as e:
        print(f"Unexpected error in quiz generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def check_quiz_answer(question_id: str, user_answer: str) -> QuizCheckResponse:
    """Check quiz answer with context retrieval and web citations."""
    if question_id not in _quiz_cache:
        raise HTTPException(status_code=404, detail="Question not found")

    try:
        cached_data = _quiz_cache[question_id]
        question_text = cached_data.get("question", "")
        topic = cached_data.get("topic", "")
        question_type = cached_data["question_type"]
        correct_answer = cached_data["correct_answer"]
        explanation = cached_data.get("explanation", "See context for details")
        citations = cached_data.get("citations", [])

        # Retrieve fresh context based on user's answer for validation
        if user_answer and user_answer.strip():
            # Create search query from question + user's answer for relevant context
            search_query = f"{question_text} {user_answer}".strip()
            if len(search_query) < 10:  # Fallback if query is too short
                search_query = f"{question_text} {topic}"

            # Get relevant context from vector database
            answer_embedding = _embed_query(search_query)
            relevant_contexts = _fetch_context(answer_embedding, QUIZ_TOP_K)

            # Combine contexts for LLM validation
            context_text = "\n\n".join([f"[{c.rank}] {c.text}" for c in relevant_contexts if c.text])
        else:
            context_text = cached_data.get("context", "")

        # Grade the answer using LLM with retrieved context
        is_correct, grade, confidence = _grade_answer(
            user_answer,
            correct_answer,
            question_type,
            question=question_text,
            context=context_text
        )

        if question_type in ['multiple_choice', 'true_false']:
            feedback = f"Your answer: '{user_answer}'. Correct answer: '{correct_answer}'"
        else:
            feedback = f"Your answer has been graded as '{grade}' with {confidence:.1%} confidence. "
            if not is_correct:
                feedback += f"The expected answer was: '{correct_answer[:100]}'"

        web_citations = []
        try:
            search_query = f"{correct_answer[:50]} {question_type.replace('_', ' ')}"
            web_citations = await _get_web_citations_for_feedback(search_query, topic)
        except Exception as e:
            print(f"Failed to get web citations: {e}")

        formatted_web_citations = [
            {
                'title': wc.title,
                'url': wc.url,
                'snippet': wc.snippet,
                'source': wc.source
            }
            for wc in web_citations
        ]

        return QuizCheckResponse(
            is_correct=is_correct,
            correct_answer=correct_answer,
            explanation=explanation,
            user_grade=grade,
            feedback=feedback,
            citations=citations,
            web_citations=formatted_web_citations,
            confidence_score=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer checking failed: {str(e)}")


def get_hardcoded_topics() -> List[str]:
    """Return list of hardcoded topics."""
    return HARDCODED_TOPICS
