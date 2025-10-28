"""
Quiz Mode API - Handles quiz generation and answer checking with web citations
"""
from __future__ import annotations

from pathlib import Path
import random
import uuid
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional, Dict
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Constants
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
QUIZ_TOP_K = 8
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
    question_type: str = 'random'
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
_openai_client: Optional[OpenAI] = None
_quiz_cache: Dict[str, Dict[str, any]] = {}


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model


def _load_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise e
    return _collection


def _load_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _embed_query(text: str) -> List[float]:
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=False, normalize_embeddings=True)
    return vec[0].tolist()


def _fetch_context(query_embedding: List[float], top_k: int) -> List[ContextItem]:
    collection = _load_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    items: List[ContextItem] = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source") if isinstance(meta, dict) else None
        page = meta.get("page") if isinstance(meta, dict) else None
        text = (doc or "").strip()
        items.append(ContextItem(rank=idx, source=source, page=page, text=text))
    return items


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


def _extract_category(context_text: str, topic: Optional[str] = None) -> str:
    """Extract category from context text."""
    categories = {
        "security": ["security", "encryption", "firewall", "vulnerability", "attack", "ddos", "vpn", "ssl", "tls"],
        "applications": ["http", "ftp", "dns", "email", "web", "browser", "smtp", "pop3", "imap"],
        "networking": ["network", "computer", "internet", "connection", "data", "information", "tcp", "ip", "routing", "osi"],
    }
    for category, keywords in categories.items():
        if any(keyword in context_text.lower() for keyword in keywords):
            return category
    return "networking"


def _generate_multiple_choice_question(question_id: str, key_sentence: str, topic_label: str, contexts: List[ContextItem], citations: List[dict], category: str) -> QuizQuestion:
    """Generate a multiple choice question with randomized distractors."""
    question_templates = {
        'security': [
            f"What is the primary function of {topic_label} in network security?",
            f"Which statement best describes how {topic_label} protects networks?",
            f"What type of threat does {topic_label} primarily address?",
            f"How does {topic_label} contribute to network defense?"
        ],
        'applications': [
            f"What is the main purpose of {topic_label} in network communication?",
            f"Which of the following best describes {topic_label}?",
            f"How does {topic_label} facilitate network operations?",
            f"What role does {topic_label} play in data transmission?"
        ],
        'networking': [
            f"What is the primary function of {topic_label}?",
            f"Which statement accurately describes {topic_label}?",
            f"How does {topic_label} support network operations?",
            f"What is the main benefit of using {topic_label}?"
        ]
    }

    template_list = question_templates.get(category, question_templates['networking'])
    question = random.choice(template_list)

    distractors = _generate_distractors(topic_label, category, key_sentence)
    correct_answer = key_sentence

    if len(correct_answer) > 100:
        correct_answer = correct_answer[:97] + "..."

    options = [correct_answer] + distractors[:3]
    random.shuffle(options)

    explanation = f"Based on the context: {correct_answer}"

    return QuizQuestion(
        id=question_id,
        type='multiple_choice',
        question=question,
        options=options,
        correct_answer=correct_answer,
        explanation=explanation,
        citations=citations
    )


def _generate_true_false_question(question_id: str, key_sentence: str, topic_label: str, contexts: List[ContextItem], citations: List[dict], category: str) -> QuizQuestion:
    """Generate a true/false question."""
    if len(key_sentence) > 80:
        statement = key_sentence[:77] + "..."
    else:
        statement = key_sentence

    question = f"True or False: {statement}"
    correct_answer = "True"
    explanation = f"This statement is derived from the context: {statement}"

    return QuizQuestion(
        id=question_id,
        type='true_false',
        question=question,
        options=None,
        correct_answer=correct_answer,
        explanation=explanation,
        citations=citations
    )


def _generate_open_ended_question(question_id: str, key_sentence: str, topic_label: str, contexts: List[ContextItem], citations: List[dict], category: str) -> QuizQuestion:
    """Generate an open-ended question."""
    question_templates = {
        'security': [
            f"Explain how {topic_label} enhances network security.",
            f"What are the key features of {topic_label} for protection?",
            f"Describe the role of {topic_label} in preventing attacks.",
            f"How does {topic_label} contribute to secure communications?"
        ],
        'applications': [
            f"Explain how {topic_label} works in network communication.",
            f"What are the main characteristics of {topic_label}?",
            f"Describe the purpose and function of {topic_label}.",
            f"How is {topic_label} typically implemented?"
        ],
        'networking': [
            f"Explain the importance of {topic_label} in networking.",
            f"What are the key aspects of {topic_label}?",
            f"Describe how {topic_label} functions.",
            f"What role does {topic_label} play in network systems?"
        ]
    }

    template_list = question_templates.get(category, question_templates['networking'])
    question = random.choice(template_list)

    correct_answer = key_sentence
    if len(correct_answer) > 200:
        correct_answer = correct_answer[:197] + "..."

    explanation = f"Key information from the context: {correct_answer}"

    return QuizQuestion(
        id=question_id,
        type='open_ended',
        question=question,
        options=None,
        correct_answer=correct_answer,
        explanation=explanation,
        citations=citations
    )


def _generate_distractors(topic_label: str, category: str, context: str) -> List[str]:
    """Generate plausible distractors for multiple choice questions."""
    base_distractors = {
        'security': [
            "It makes networks completely immune to all attacks",
            "It eliminates the need for passwords",
            "It only works with older security systems",
            "It reduces network speed significantly",
            "It conflicts with modern security protocols"
        ],
        'applications': [
            "It only works with specific operating systems",
            "It requires special hardware to function",
            "It is primarily used for entertainment",
            "It slows down data transmission",
            "It is incompatible with modern protocols"
        ],
        'networking': [
            "It only works with wired connections",
            "It requires extensive manual configuration",
            "It is unrelated to network reliability",
            "It decreases network performance",
            "It conflicts with standard networking protocols"
        ]
    }

    category_distractors = base_distractors.get(category, base_distractors['networking'])
    return random.sample(category_distractors, min(len(category_distractors), 4))


def _generate_question_from_context(contexts: List[ContextItem], question_type: str, topic: Optional[str] = None) -> QuizQuestion:
    """Generate a quiz question from database context with randomization."""
    if not contexts:
        contexts = [ContextItem(rank=1, source=None, page=None, text=f"Key facts about {topic or 'networking'}.")]

    # Randomly select a context to use
    primary_context = random.choice(contexts)
    base_text = (primary_context.text or "").strip()

    # Extract random sentence from context
    sentences = [s.strip() for s in base_text.split('.') if s.strip() and len(s.strip()) > 20]
    key_sentence = random.choice(sentences) if sentences else f"{topic or 'networking'} is important."

    category = _extract_category(base_text, topic)

    citations = [
        {"source": c.source or "context", "page": c.page, "rank": c.rank}
        for c in contexts[:3] if c.text and c.text.strip()
    ]

    question_id = str(uuid.uuid4())
    topic_label = (topic or category).title()

    if question_type == 'multiple_choice':
        return _generate_multiple_choice_question(
            question_id, key_sentence, topic_label, contexts, citations, category
        )
    elif question_type == 'true_false':
        return _generate_true_false_question(
            question_id, key_sentence, topic_label, contexts, citations, category
        )
    elif question_type == 'open_ended':
        return _generate_open_ended_question(
            question_id, key_sentence, topic_label, contexts, citations, category
        )
    else:
        return _generate_multiple_choice_question(
            question_id, key_sentence, topic_label, contexts, citations, category
        )


def _grade_answer(user_answer: str, correct_answer: str, question_type: str) -> tuple[bool, str, float]:
    """Enhanced answer grading with confidence scoring."""
    if not user_answer or not user_answer.strip():
        return False, 'F', 0.0

    user_text = user_answer.lower().strip()
    correct_text = correct_answer.lower().strip()

    if question_type in ['multiple_choice', 'true_false']:
        if user_text == correct_text:
            return True, 'A', 1.0
        else:
            return False, 'F', 0.0
    else:  # open_ended
        similarity = SequenceMatcher(None, user_text, correct_text).ratio()

        correct_words = set(re.findall(r'\b\w{4,}\b', correct_text))
        user_words = set(re.findall(r'\b\w{4,}\b', user_text))
        key_term_overlap = len(correct_words.intersection(user_words)) / len(correct_words) if correct_words else 0

        combined_score = (similarity * 0.7) + (key_term_overlap * 0.3)

        if combined_score >= 0.8:
            return True, 'A', combined_score
        elif combined_score >= 0.65:
            return True, 'B', combined_score
        elif combined_score >= 0.5:
            return True, 'C', combined_score
        elif combined_score >= 0.3:
            return False, 'D', combined_score
        else:
            return False, 'F', combined_score


async def _get_web_citations_for_feedback(question: str, topic: str) -> List[WebCitation]:
    """Get web citations for quiz feedback."""
    search_query = f"{topic} {question}" if topic else question
    return await _search_web(search_query, max_results=2)


def generate_quiz(topic: Optional[str], question_type: str, count: int) -> QuizResponse:
    """Generate quiz questions - randomly or topic-specific."""
    try:
        # If no topic provided, randomly select from hardcoded topics
        if not topic or not topic.strip():
            topic_query = random.choice(HARDCODED_TOPICS)
        else:
            topic_query = topic.strip()

        contexts = _fetch_context(_embed_query(topic_query), QUIZ_TOP_K)

        if not contexts:
            raise HTTPException(status_code=404, detail="No context for quiz generation")

        questions = []
        num_questions = max(MIN_QUESTIONS, min(count, MAX_QUESTIONS))
        
        for _ in range(num_questions):
            # Randomize question type if 'random' is selected
            q_type = question_type if question_type != 'random' else random.choice(['multiple_choice', 'true_false', 'open_ended'])
            
            # Generate question with randomization
            q = _generate_question_from_context(contexts, q_type, topic_query)
            
            # Cache question data for checking later
            _quiz_cache[q.id] = {
                "correct_answer": q.correct_answer,
                "question_type": q.type,
                "explanation": q.explanation,
                "citations": q.citations,
                "topic": topic_query
            }
            questions.append(q)

        return QuizResponse(total_questions=len(questions), questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def check_quiz_answer(question_id: str, user_answer: str) -> QuizCheckResponse:
    """Check quiz answer with web citations."""
    if question_id not in _quiz_cache:
        raise HTTPException(status_code=404, detail="Question not found")

    try:
        cached_data = _quiz_cache[question_id]
        correct_answer = cached_data["correct_answer"]
        question_type = cached_data["question_type"]
        explanation = cached_data.get("explanation", "See context for details")
        citations = cached_data.get("citations", [])

        is_correct, grade, confidence = _grade_answer(user_answer, correct_answer, question_type)

        if question_type in ['multiple_choice', 'true_false']:
            feedback = f"Your answer: '{user_answer}'. Correct answer: '{correct_answer}'"
        else:
            feedback = f"Your answer has been graded as '{grade}' with {confidence:.1%} confidence. "
            if not is_correct:
                feedback += f"The expected answer was: '{correct_answer[:100]}'"

        web_citations = []
        try:
            search_query = f"{correct_answer[:50]} {question_type.replace('_', ' ')}"
            web_citations = await _get_web_citations_for_feedback(search_query, cached_data.get('topic', ''))
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
