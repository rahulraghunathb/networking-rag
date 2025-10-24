const API_URL = "http://127.0.0.1:8000"
const chat = document.getElementById("chat")
const promptEl = document.getElementById("prompt")
const sendBtn = document.getElementById("send")
const newChatBtn = document.getElementById("new-chat")

function appendBubble(role, text) {
  const wrap = document.createElement("div")
  wrap.className = `bubble ${role}`
  const avatar = document.createElement("div")
  avatar.className = "avatar"
  avatar.textContent = role === "user" ? "You" : "AI"
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

sendBtn.addEventListener("click", async () => {
  const q = promptEl.value.trim()
  if (!q) return
  appendBubble("user", q)
  promptEl.value = ""
  try {
    const resp = await ask(q)
    renderAnswer(resp)
  } catch (e) {
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
