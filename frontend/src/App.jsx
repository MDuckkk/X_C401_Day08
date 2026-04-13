import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';
import AdminPage from './AdminPage';

// ─── Highlighted Source Component ─────────────────────────────────────────────

const STOP_WORDS = new Set([
  'the','a','an','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','will','would','shall','should','may','might','must','can',
  'could','to','of','in','for','on','with','at','by','from','as','into','about',
  'between','through','during','before','after','above','below','up','down',
  'out','off','over','under','again','further','then','once','here','there',
  'when','where','why','how','all','each','every','both','few','more','most',
  'other','some','such','no','nor','not','only','own','same','so','than','too',
  'very','just','because','but','and','or','if','that','this','it','its','they',
  'their','them','these','those','what','which','who','whom','your','you','we',
  'our','my','me','he','she','his','her','i','also','use','using','please',
  'make','sure','check','need','get','go','see','like','know','want','take',
  // Vietnamese stop words
  'của','và','trong','cho','các','với','không','có','được','những','là','như','từ',
  'một','đến','để','về','tại','này','khi','bởi','nên','nếu','thì','đã','sẽ','đang',
  'bạn','vui','lòng','liên','hệ','người','dùng','mới','rất','nhiều','hay','hoặc'
]);

function HighlightedSource({ content, answerText }) {
  if (!answerText || !content) return content;

  // Extract meaningful keywords from the answer (>= 3 chars, not stop words, support Vietnamese accents)
  const answerKeywords = new Set(
    answerText.toLowerCase()
      .replace(/[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length >= 3 && !STOP_WORDS.has(w))
  );

  if (answerKeywords.size === 0) return content;

  // Split content into sentences/clauses. We keep the delimiters so we can rebuild exactly.
  // Regex matches characters up to a punctuation mark + space/newline, OR end of string.
  const sentenceRegex = /([^.?!:\n]+[.?!:\n]+|[^.?!:\n]+$)/g;
  const parts = content.match(sentenceRegex) || [content];

  return parts.map((part, i) => {
    // Count how many answerKeywords are in this sentence
    const partWords = part.toLowerCase()
      .replace(/[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]/g, ' ')
      .split(/\s+/);
      
    let matchCount = 0;
    for (const pw of partWords) {
      if (answerKeywords.has(pw)) matchCount++;
    }

    // Require at least 2 strong keyword matches to highlight a sentence
    // For longer sentences, scale it slightly up to 3 max.
    const threshold = Math.min(3, Math.max(2, Math.floor(partWords.length * 0.15)));
    
    if (matchCount >= threshold) {
      return <mark key={i} className="source-highlight">{part}</mark>;
    }
    return part;
  });
}

// Generate or retrieve a persistent user ID
function getUserId() {
  let userId = localStorage.getItem('it_support_user_id');
  if (!userId) {
    userId = 'user-' + crypto.randomUUID();
    localStorage.setItem('it_support_user_id', userId);
  }
  return userId;
}

// Generate a new thread ID for each chat session
function newThreadId() {
  return 'thread-' + crypto.randomUUID();
}

// ─── Thread Storage (localStorage) ────────────────────────────────────────────

function loadThreads() {
  try {
    const raw = localStorage.getItem('it_support_threads');
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveThreads(threads) {
  localStorage.setItem('it_support_threads', JSON.stringify(threads));
}

function upsertThread(threadId, title) {
  const threads = loadThreads();
  const existing = threads.find(t => t.id === threadId);
  if (existing) {
    existing.title = title || existing.title;
    existing.updatedAt = Date.now();
  } else {
    threads.unshift({ id: threadId, title: title || 'New Chat', createdAt: Date.now(), updatedAt: Date.now() });
  }
  saveThreads(threads);
  return loadThreads();
}

function deleteThread(threadId) {
  const threads = loadThreads().filter(t => t.id !== threadId);
  saveThreads(threads);
  localStorage.removeItem(`it_thread_msgs_${threadId}`);
  return threads;
}

function saveThreadMessages(threadId, messages) {
  // Only save non-loading, non-streaming messages
  const toSave = messages.filter(m => !m.isLoading && !m.isStreaming);
  localStorage.setItem(`it_thread_msgs_${threadId}`, JSON.stringify(toSave));
}

function loadThreadMessages(threadId) {
  try {
    const raw = localStorage.getItem(`it_thread_msgs_${threadId}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

// ─── App Component ────────────────────────────────────────────────────────────

function App() {
  const [currentPage, setCurrentPage] = useState(window.location.pathname === '/admin' ? 'admin' : 'chat');

  useEffect(() => {
    const handlePopState = () => {
      setCurrentPage(window.location.pathname === '/admin' ? 'admin' : 'chat');
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const navigateTo = (path) => {
    window.history.pushState({}, '', path);
    setCurrentPage(path === '/admin' ? 'admin' : 'chat');
  };
  const [messages, setMessages] = useState([
    { id: -1, text: "", sender: 'bot', sources: [], isLoading: true }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [statusText, setStatusText] = useState("");
  const messagesEndRef = useRef(null);

  // Sidebar
  const [threads, setThreads] = useState(() => loadThreads());

  // Persistent user ID + per-session thread ID
  const [userId] = useState(() => getUserId());
  const [threadId, setThreadId] = useState(() => newThreadId());

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, statusText]);

  // Auto-save messages to localStorage when they change
  useEffect(() => {
    // Only save if there's actual user conversation (not just the greeting)
    const hasUserMessages = messages.some(m => m.sender === 'user');
    if (hasUserMessages && modelReady) {
      saveThreadMessages(threadId, messages);
    }
  }, [messages, threadId, modelReady]);

  // Load greeting after initial delay to simulate model initialization
  useEffect(() => {
    const timer = setTimeout(() => {
      setMessages([
        { id: 1, text: "Hello! I'm your IT Support Assistant, here to provide quick answers to your technical questions. Please describe your issue.", sender: 'bot', sources: [] }
      ]);
      setModelReady(true);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  const handleNewChat = () => {
    const newId = newThreadId();
    setThreadId(newId);
    setMessages([
      { id: 1, text: "Hello! I'm your IT Support Assistant, here to provide quick answers to your technical questions. Please describe your issue.", sender: 'bot', sources: [] }
    ]);
  };

  // Load a past thread from localStorage
  const handleLoadThread = (tid) => {
    setThreadId(tid);
    const saved = loadThreadMessages(tid);
    if (saved && saved.length > 0) {
      setMessages(saved);
    } else {
      setMessages([
        { id: 1, text: "Hello! I'm your IT Support Assistant, here to provide quick answers to your technical questions. Please describe your issue.", sender: 'bot', sources: [] }
      ]);
    }
  };

  const handleDeleteThread = (e, tid) => {
    e.stopPropagation();
    const updated = deleteThread(tid);
    setThreads(updated);
    // If deleting the active thread, start a new chat
    if (tid === threadId) {
      handleNewChat();
    }
  };

  const handleSend = async () => {
    const query = input.trim();
    if (!query) return;

    // Save/update thread in sidebar list (use first user message as title)
    const isFirstMessage = messages.filter(m => m.sender === 'user').length === 0;
    const title = isFirstMessage ? query.slice(0, 50) + (query.length > 50 ? '…' : '') : undefined;
    setThreads(upsertThread(threadId, title));

    // Add user message
    const userMessage = { id: Date.now(), text: query, sender: 'user', sources: [] };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setStatusText("");

    // Add a bot message that we'll stream into
    const botId = Date.now() + 1;
    setMessages(prev => [...prev, { id: botId, text: "", sender: 'bot', sources: [], isStreaming: true }]);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, user_id: userId, thread_id: threadId })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Failed to fetch response.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let sources = [];
      let isFromRag = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.type === "status") {
              setStatusText(event.content);
            } else if (event.type === "token") {
              setStatusText(""); // clear status once tokens start
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === botId
                    ? { ...msg, text: msg.text + event.content }
                    : msg
                )
              );
            } else if (event.type === "done") {
              sources = event.sources || [];
              isFromRag = event.is_from_rag || false;
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === botId
                    ? { ...msg, sources, is_from_rag: isFromRag, isStreaming: false }
                    : msg
                )
              );
            } else if (event.type === "error") {
              throw new Error(event.content);
            }
          } catch (parseErr) {
            // skip malformed SSE lines
          }
        }
      }

      // Update thread timestamp
      setThreads(upsertThread(threadId));
      setStatusText("");
    } catch (error) {
      console.error("Error:", error);
      setStatusText("");
      setMessages(prev =>
        prev.map(msg =>
          msg.id === botId
            ? { id: botId, text: `Error: ${error.message}`, sender: 'system-error', sources: [], isStreaming: false }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
      setStatusText("");
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSend();
    }
  };

  const formatTime = (timestamp) => {
    const d = new Date(timestamp);
    const now = new Date();
    const diffMs = now - d;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString();
  };

  return (
    <>
      {currentPage === 'chat' ? (
        <div className="app-layout">
          {/* Chat history sidebar — always visible */}
          <aside className="chat-sidebar">
            <div className="sidebar-header">
              <h2>Chat History</h2>
            </div>
            <button className="sidebar-new-chat-btn" onClick={handleNewChat}>
              + New Chat
            </button>
            <div className="sidebar-thread-list">
              {threads.length === 0 ? (
                <div className="sidebar-empty">No past conversations</div>
              ) : (
                threads
                  .sort((a, b) => b.updatedAt - a.updatedAt)
                  .map(t => (
                    <div
                      key={t.id}
                      className={`sidebar-thread-item ${t.id === threadId ? 'active' : ''}`}
                      onClick={() => handleLoadThread(t.id)}
                    >
                      <div className="thread-info">
                        <span className="thread-title">{t.title}</span>
                        <span className="thread-time">{formatTime(t.updatedAt)}</span>
                      </div>
                      <button
                        className="thread-delete-btn"
                        onClick={(e) => handleDeleteThread(e, t.id)}
                        title="Delete conversation"
                      >
                        🗑
                      </button>
                    </div>
                  ))
              )}
            </div>
          </aside>

          {/* Main chat container */}
          <div className="chatbot-container">
            <header className="chatbot-header">
              <div className="header-content">
                <div>
                  <h1>IT Support Agent</h1>
                  <p>Powered by LangGraph & OpenAI</p>
                </div>
                <div className="header-actions">
                  <button className="admin-link" onClick={() => navigateTo('/admin')}>⚙️ Admin</button>
                </div>
              </div>
            </header>

            <main className="chat-history">
              {messages
                .filter(msg => !(msg.isStreaming && !msg.text))
                .map((msg) => (
                <div key={msg.id} className={`message ${msg.sender}`}>
                  <div className={`message-bubble ${msg.isLoading ? 'loading' : ''}`}>
                    {msg.isLoading ? (
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    ) : (
                      <>
                        <div className="markdown-body">
                          <ReactMarkdown>{msg.text}</ReactMarkdown>
                        </div>
                        {msg.is_from_rag && msg.sources && msg.sources.length > 0 && !msg.isStreaming && (
                          <div className="sources-container">
                            <div className="sources-label">📚 Sources:</div>
                            {msg.sources.map((source, idx) => (
                              <div
                                key={idx}
                                className="source-item clickable"
                                onClick={() => {
                                  setMessages(prev =>
                                    prev.map(m =>
                                      m.id === msg.id
                                        ? {
                                            ...m,
                                            sources: m.sources.map((s, si) =>
                                              si === idx ? { ...s, expanded: !s.expanded } : s
                                            ),
                                          }
                                        : m
                                    )
                                  );
                                }}
                              >
                                <div className="source-header">
                                  <span className="source-id">{source.source}</span>
                                  <span className="source-toggle">{source.expanded ? '▼' : '▶'}</span>
                                </div>
                                {source.expanded && (
                                  <div className="source-content">
                                    <HighlightedSource content={source.content} answerText={msg.text} />
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
              {statusText && (
                <div className="message bot">
                  <div className="status-indicator">
                    <div className="status-spinner"></div>
                    <span>{statusText}</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </main>

            <footer className="chat-input-area">
              <input
                type="text"
                className="chat-input"
                placeholder={modelReady ? "Type your IT question here..." : "Loading model..."}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading || !modelReady}
                autoFocus
              />
              <button
                className="send-button"
                onClick={handleSend}
                disabled={isLoading || !modelReady}
              >
                {!modelReady ? "Loading..." : isLoading ? "Thinking..." : "Send"}
              </button>
            </footer>
          </div>
        </div>
      ) : (
        <AdminPage onBack={() => navigateTo('/')} />
      )}
    </>
  );
}

export default App;
