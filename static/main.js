// API URL
const API_URL = "http://127.0.0.1:8000"
const chat = document.getElementById("chat")
const promptEl = document.getElementById("prompt")
const sendBtn = document.getElementById("send")
const newChatBtn = document.getElementById("new-chat")
const exampleBtns = document.querySelectorAll(".example-btn")

const modeToggle = document.getElementById("mode-toggle")
const chatContainer = document.getElementById("chat-container")
const quizContainer = document.getElementById("quiz-container")
const examples = document.getElementById("examples")
const footer = document.querySelector(".footer")
const generateQuizBtn = document.getElementById("generate-quiz")
const quizQuestionsEl = document.getElementById("quiz-questions")
const quizFeedbackEl = document.getElementById("quiz-feedback")
const quizTypeSelect = document.getElementById("quiz-type")
const quizTopicInput = document.getElementById("quiz-topic")

let quizMode = false
let currentQuestions = []

const modeSelect = document.getElementById("mode-select")
const loadingQuiz = document.querySelector(".loading-quiz")

// Handle mode selection
modeSelect.addEventListener("change", () => {
  const mode = modeSelect.value
  if (mode === "qa") {
    chatContainer.classList.remove("hidden")
    quizContainer.classList.add("hidden")
    examples.classList.remove("hidden")
    footer.classList.remove("hidden")
  } else {
    chatContainer.classList.add("hidden")
    quizContainer.classList.remove("hidden")
    examples.classList.add("hidden")
    footer.classList.add("hidden")
  }
})

const quizCountInput = document.getElementById("quiz-count")
const generationTypeRadios = document.querySelectorAll(
  'input[name="generation-type"]'
)
const topicSection = document.getElementById("topic-section")

// Fetch and populate hardcoded topics
async function loadTopics() {
  try {
    const resp = await fetch(`${API_URL}/quiz/topics`)
    const data = await resp.json()

    quizTopicInput.innerHTML = '<option value="">-- Select a Topic --</option>'
    data.topics.forEach((topic) => {
      const option = document.createElement("option")
      option.value = topic
      option.textContent = topic
      quizTopicInput.appendChild(option)
    })
  } catch (e) {
    console.error("Error loading topics:", e)
    quizTopicInput.innerHTML = '<option value="">Error loading topics</option>'
  }
}

// Load topics on page load
loadTopics()

// Handle generation type toggle
generationTypeRadios.forEach((radio) => {
  radio.addEventListener("change", (e) => {
    if (e.target.value === "topic") {
      topicSection.style.display = "flex"
    } else {
      topicSection.style.display = "none"
      quizTopicInput.value = "" // Clear topic when switching to random
    }
  })
})

async function generateQuiz() {
  loadingQuiz.classList.remove("hidden")
  generateQuizBtn.disabled = true

  const type = quizTypeSelect.value
  const generationType = document.querySelector(
    'input[name="generation-type"]:checked'
  ).value
  const topic = generationType === "topic" ? quizTopicInput.value : null
  const count = parseInt(quizCountInput.value) || 3

  // Validate topic selection if topic-specific is chosen
  if (generationType === "topic" && !topic) {
    alert("Please select a topic for topic-specific questions")
    loadingQuiz.classList.add("hidden")
    generateQuizBtn.disabled = false
    return
  }

  const payload = { topic: topic, question_type: type, count: count }

  try {
    const resp = await fetch(`${API_URL}/quiz/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
    const data = await resp.json()
    currentQuestions = data.questions
    renderQuizQuestions(currentQuestions)
  } catch (e) {
    console.error("Error generating quiz:", e)
    alert("Error generating quiz. Please try again.")
  } finally {
    loadingQuiz.classList.add("hidden")
    generateQuizBtn.disabled = false
  }
}

function renderQuizQuestions(questions) {
  quizQuestionsEl.innerHTML = ""
  quizFeedbackEl.innerHTML = "" // Clear global feedback
  quizFeedbackEl.classList.add("hidden")
  
  questions.forEach((q, index) => {
    const qDiv = document.createElement("div")
    qDiv.className = "quiz-question"
    qDiv.setAttribute("data-qid", q.id)
    
    // Create question header
    const questionHeader = document.createElement("h4")
    questionHeader.textContent = `${index + 1}. ${q.question}`
    qDiv.appendChild(questionHeader)
    
    // Create options container
    if (q.options && q.options.length > 0) {
      const optionsDiv = document.createElement("div")
      optionsDiv.className = "quiz-options"
      
      // Add each option as a button
      q.options.forEach(opt => {
        const optionBtn = document.createElement("button")
        optionBtn.className = "quiz-option"
        optionBtn.setAttribute("data-answer", opt)
        optionBtn.setAttribute("data-qid", q.id)
        optionBtn.textContent = opt
        optionsDiv.appendChild(optionBtn)
      })
      
      qDiv.appendChild(optionsDiv)
    } else {
      // For open-ended questions
      const textarea = document.createElement("textarea")
      textarea.className = "quiz-textarea"
      textarea.setAttribute("data-qid", q.id)
      textarea.setAttribute("placeholder", "Your answer...")
      qDiv.appendChild(textarea)
    }
    
    // Add submit button
    const submitBtn = document.createElement("button")
    submitBtn.className = "btn submit-answer"
    submitBtn.setAttribute("data-qid", q.id)
    submitBtn.textContent = "Submit Answer"
    qDiv.appendChild(submitBtn)
    
    // Add feedback container
    const feedbackDiv = document.createElement("div")
    feedbackDiv.className = "quiz-feedback-inline hidden"
    feedbackDiv.id = `feedback-${q.id}`
    qDiv.appendChild(feedbackDiv)
    
    quizQuestionsEl.appendChild(qDiv)
  })
}

async function submitAnswer(qid, answer) {
  try {
    const resp = await fetch(`${API_URL}/quiz/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question_id: qid, user_answer: answer }),
    })
    const data = await resp.json()
    renderQuizFeedbackInline(qid, data)
  } catch (e) {
    console.error("Error checking answer:", e)
  }
}

function renderQuizFeedbackInline(qid, feedback) {
  const feedbackDiv = document.getElementById(`feedback-${qid}`)
  if (!feedbackDiv) return
  
  feedbackDiv.classList.remove("hidden", "correct", "incorrect")
  feedbackDiv.className = `quiz-feedback-inline ${
    feedback.is_correct ? "correct" : "incorrect"
  }`
  feedbackDiv.innerHTML = `
    <h4>Quiz Feedback</h4>
    <div class="feedback-content">
      <p><strong>Correct Answer:</strong> ${feedback.correct_answer}</p>
      <p><strong>Your Grade:</strong> ${feedback.user_grade} (${(
    feedback.confidence_score * 100
  ).toFixed(1)}% confidence)</p>
      <p><strong>Feedback:</strong> ${feedback.feedback}</p>
      <p><strong>Explanation:</strong> ${feedback.explanation}</p>

      ${
        feedback.citations && feedback.citations.length > 0
          ? `
        <div class="citations-section">
          <h5>Database Citations:</h5>
          ${feedback.citations
            .map(
              (c) =>
                `<div class="citation">
               <span class="citation-source">[${c.source || "unknown"}${
                  c.page ? " · p" + c.page : ""
                }]</span>
             </div>`
            )
            .join("")}
        </div>
      `
          : ""
      }

      ${
        feedback.web_citations && feedback.web_citations.length > 0
          ? `
        <div class="web-citations-section">
          <h5>Additional References:</h5>
          ${feedback.web_citations
            .map(
              (wc) =>
                `<div class="web-citation">
               <h6><a href="${wc.url}" target="_blank" rel="noopener noreferrer">${wc.title}</a></h6>
               <p>${wc.snippet}</p>
               <small class="citation-source">Source: ${wc.source}</small>
             </div>`
            )
            .join("")}
        </div>
      `
          : ""
      }
    </div>
  `
}

generateQuizBtn.addEventListener("click", generateQuiz)

quizQuestionsEl.addEventListener("click", (e) => {
  if (e.target.classList.contains("quiz-option")) {
    // Select the option (highlight it) instead of auto-submitting
    const qid = e.target.getAttribute("data-qid")
    const questionDiv = document.querySelector(`.quiz-question[data-qid="${qid}"]`)
    
    // Remove selection from all options in this question
    questionDiv.querySelectorAll(".quiz-option").forEach(opt => {
      opt.classList.remove("selected")
    })
    
    // Mark this option as selected
    e.target.classList.add("selected")
    
  } else if (e.target.classList.contains("submit-answer")) {
    e.preventDefault()
    const qid = e.target.getAttribute("data-qid")
    const questionDiv = document.querySelector(`.quiz-question[data-qid="${qid}"]`)
    
    // Check if it's MCQ or open-ended
    const selectedOption = questionDiv.querySelector(".quiz-option.selected")
    const textarea = questionDiv.querySelector(".quiz-textarea")
    
    let answer = null
    if (selectedOption) {
      answer = selectedOption.getAttribute("data-answer")
    } else if (textarea) {
      answer = textarea.value.trim()
    }
    
    if (answer) {
      submitAnswer(qid, answer)
    } else {
      alert("Please select or enter an answer before submitting")
    }
  }
})

exampleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const question = btn.getAttribute("data-question")
    promptEl.value = question
    promptEl.focus()
    sendBtn.click() // Auto-submit
  })
})

function parseMarkdown(text) {
  // Escape HTML to prevent XSS
  const escapeHtml = (str) => {
    const div = document.createElement('div')
    div.textContent = str
    return div.innerHTML
  }
  
  // Split by ** to handle bold text
  let result = ''
  let inBold = false
  const parts = text.split('**')
  
  parts.forEach((part, index) => {
    if (index === 0) {
      result += escapeHtml(part)
    } else {
      if (inBold) {
        result += '</strong>' + escapeHtml(part)
      } else {
        result += '<strong>' + escapeHtml(part)
      }
      inBold = !inBold
    }
  })
  
  // Preserve line breaks
  result = result.replace(/\n/g, '<br>')
  
  return result
}

function appendBubble(role, text) {
  const wrap = document.createElement("div")
  wrap.className = `bubble ${role}`
  const avatar = document.createElement("div")
  avatar.className = "avatar"
  avatar.innerHTML =
    role === "user"
      ? `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/></svg>`
      : `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M9 12c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-6 6c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3z" fill="currentColor"/></svg>`
  const content = document.createElement("div")
  content.className = "content"
  
  // Parse markdown for assistant responses
  if (role === "assistant") {
    content.innerHTML = parseMarkdown(text)
  } else {
    content.textContent = text
  }
  
  wrap.appendChild(avatar)
  wrap.appendChild(content)
  chat.appendChild(wrap)
  chat.scrollTop = chat.scrollHeight
}

async function ask(question) {
  const body = { question, top_k: 4 }
  console.log("Sending Q&A request:", body)
  
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  
  console.log("Q&A response status:", res.status)
  
  if (!res.ok) {
    const txt = await res.text()
    console.error("Q&A error response:", txt)
    throw new Error(txt || `HTTP ${res.status}`)
  }
  
  const data = await res.json()
  console.log("Q&A parsed response:", data)
  return data
}

function renderAnswer(resp) {
  console.log("Q&A Response:", resp)
  const { snippet, results, citations } = resp
  
  if (!snippet) {
    console.error("No snippet in response:", resp)
    appendBubble("assistant", "Error: No answer received")
    return
  }
  
  appendBubble("assistant", snippet)
  const last = chat.lastElementChild.querySelector(".content")
  if (Array.isArray(citations) && citations.length) {
    const mapDiv = document.createElement("div")
    mapDiv.className = "sources"
    mapDiv.innerHTML = citations
      .map(
        (c) =>
          `<span class="badge" data-rank="${c.index}">[${c.index}] ${
            c.source || "unknown"
          }${c.page ? " · p" + c.page : ""}</span>`
      )
      .join(" ")
    last.appendChild(mapDiv)
  } else if (results && results.length) {
    // Fallback to results if citations missing
    const div = document.createElement("div")
    div.className = "sources"
    div.innerHTML = results
      .map(
        (r, i) =>
          `<span class="badge" data-rank="${i + 1}">[${i + 1}] ${
            r.source || "unknown"
          }${r.page ? " · p" + r.page : ""}</span>`
      )
      .join(" ")
    last.appendChild(div)
  }
}

sendBtn.addEventListener("click", async () => {
  const q = promptEl.value.trim()
  if (!q) return
  appendBubble("user", q)
  promptEl.value = ""

  const thinkingWrap = document.createElement("div")
  thinkingWrap.className = "bubble assistant"
  const thinkingAvatar = document.createElement("div")
  thinkingAvatar.className = "avatar"
  thinkingAvatar.innerHTML =
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M9 12c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-6 6c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3z" fill="currentColor"/></svg>'
  const thinkingContent = document.createElement("div")
  thinkingContent.className = "content thinking"
  thinkingContent.innerHTML =
    '<div class="thinking-indicator"><div class="thinking-spinner"></div><span class="thinking-text">Thinking...</span></div>'
  thinkingWrap.appendChild(thinkingAvatar)
  thinkingWrap.appendChild(thinkingContent)
  chat.appendChild(thinkingWrap)
  chat.scrollTop = chat.scrollHeight

  try {
    const resp = await ask(q)
    chat.removeChild(thinkingWrap)
    renderAnswer(resp)
  } catch (e) {
    chat.removeChild(thinkingWrap)
    appendBubble("assistant", `Error: ${e.message}`)
  }
})

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    sendBtn.click()
  }
})

if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    chat.innerHTML = ""
    const meta = document.getElementById("meta")
    if (meta) meta.textContent = ""
    promptEl.value = ""
    promptEl.focus()
  })
}
