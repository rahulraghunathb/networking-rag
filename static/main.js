const API_URL = "http://127.0.0.1:8000"
const chat = document.getElementById("chat")
const promptEl = document.getElementById("prompt")
const sendBtn = document.getElementById("send")
const newChatBtn = document.getElementById("new-chat")
const exampleBtns = document.querySelectorAll(".example-btn")

// Handle example button clicks
exampleBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const question = btn.getAttribute("data-question")
    promptEl.value = question
    promptEl.focus()
    sendBtn.click() // Auto-submit
  })
})

function appendBubble(role, text) {
  const wrap = document.createElement("div")
  wrap.className = `bubble ${role}`
  const avatar = document.createElement("div")
  avatar.className = "avatar"
  avatar.innerHTML = role === 'user' ?
    `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" fill="currentColor"/></svg>` :
    `<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M9 12c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-6 6c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3z" fill="currentColor"/></svg>`; 
  const content = document.createElement("div")
  content.className = "content"
  content.textContent = text
  wrap.appendChild(avatar)
  wrap.appendChild(content)
  chat.appendChild(wrap)
  chat.scrollTop = chat.scrollHeight
}

async function ask(question) {
  const body = { question, top_k: 4 }
  const res = await fetch(`${API_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const txt = await res.text()
    throw new Error(txt || `HTTP ${res.status}`)
  }
  return await res.json()
}

function renderAnswer(resp) {
  const { answer, results, citations } = resp
  appendBubble("assistant", answer)
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

sendBtn.addEventListener('click', async () => {
  const q = promptEl.value.trim()
  if (!q) return
  appendBubble('user', q)
  promptEl.value = ''
  
  // Show thinking indicator
  const thinkingWrap = document.createElement('div')
  thinkingWrap.className = 'bubble assistant'
  const thinkingAvatar = document.createElement('div')
  thinkingAvatar.className = 'avatar'
  thinkingAvatar.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M9 12c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm-6 6c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3zm6 0c1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3 1.34 3 3 3z" fill="currentColor"/></svg>'
  const thinkingContent = document.createElement('div')
  thinkingContent.className = 'content thinking'
  thinkingContent.innerHTML = '<div class="thinking-indicator"><div class="thinking-spinner"></div><span class="thinking-text">Thinking...</span></div>'
  thinkingWrap.appendChild(thinkingAvatar)
  thinkingWrap.appendChild(thinkingContent)
  chat.appendChild(thinkingWrap)
  chat.scrollTop = chat.scrollHeight
  
  try {
    const resp = await ask(q)
    // Remove thinking indicator
    chat.removeChild(thinkingWrap)
    renderAnswer(resp)
  } catch (e) {
    // Remove thinking indicator
    chat.removeChild(thinkingWrap)
    appendBubble('assistant', `Error: ${e.message}`)
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
