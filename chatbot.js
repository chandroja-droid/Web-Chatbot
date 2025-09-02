// ==== Configuration ====
const API_BASE = "http://localhost:8000";
const REQUEST_TIMEOUT = 30000;

// ==== State Management ====
let chatState = {
    isMinimized: false,
    uploadedFiles: [],
    isProcessing: false,
    isConnected: false,
    backendHealth: 'unknown'
};

// ==== DOM References ====
function getElement(id) {
    return document.getElementById(id);
}

// ==== Initialize Chatbot ====
function initializeChatbot() {
    const root = getElement('chatbot');
    if (!root) {
        console.error('Chatbot container not found!');
        return;
    }

    root.innerHTML = `
        <div class="cb-header" id="toggle-chat">
            <div class="cb-title">💬 Document Assistant</div>
            <div class="cb-actions">
                <span id="connection-status" class="status-indicator" title="Checking connection...">●</span>
                <button id="minimize-chat" title="Minimize">−</button>
                <button id="close-chat" title="Close">×</button>
            </div>
        </div>
        
        <div class="cb-tools" id="tools">
            <label class="cb-btn" for="file-input">
                <span>📎 Upload Files</span>
            </label>
            <input id="file-input" type="file" accept=".pdf,.docx,.doc,.csv,.txt,.xlsx,.xls" multiple />
        </div>
        
        <div id="messages" class="cb-messages"></div>
        
        <form id="ask-form" class="cb-form">
            <input id="question" type="text" placeholder="Ask about your documents..." autocomplete="off" />
            <button type="submit" id="send-btn">
                <span>Send</span>
                <span>→</span>
            </button>
        </form>
    `;

    // Initialize event listeners
    initializeEventListeners();
    
    // Show welcome messages
    addMessage("Hello! I can help you analyze your documents. 📚", "bot");
    addMessage("Upload files (PDF, DOCX, CSV, etc.) and ask me questions about your content.", "bot");
    
    // Check backend connection
    checkBackendHealth();
}

// ==== Event Listeners ====
function initializeEventListeners() {
    // Form submission
    const askForm = getElement('ask-form');
    if (askForm) {
        askForm.addEventListener('submit', handleAskFormSubmit);
    }

    // File input
    const fileInput = getElement('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', handleFileInputChange);
    }

    // Minimize button
    const minimizeBtn = getElement('minimize-chat');
    if (minimizeBtn) {
        minimizeBtn.addEventListener('click', () => {
            chatState.isMinimized = !chatState.isMinimized;
            updateChatVisibility();
        });
    }

    // Close button
    const closeBtn = getElement('close-chat');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            getElement('chatbot').style.display = 'none';
            showNotification("Chat closed. Refresh page to reopen.", "info");
        });
    }
}

function updateChatVisibility() {
    const tools = getElement('tools');
    const messages = getElement('messages');
    const askForm = getElement('ask-form');
    const chatbot = getElement('chatbot');

    if (tools) tools.style.display = chatState.isMinimized ? 'none' : 'flex';
    if (messages) messages.style.display = chatState.isMinimized ? 'none' : 'flex';
    if (askForm) askForm.style.display = chatState.isMinimized ? 'none' : 'flex';
    if (chatbot) chatbot.classList.toggle('minimized', chatState.isMinimized);
}

// ==== UI Functions ====
function addMessage(text, role = "bot") {
    const messagesContainer = getElement('messages');
    if (!messagesContainer) return;

    const div = document.createElement("div");
    div.className = `cb-msg ${role}`;
    div.textContent = text;
    messagesContainer.appendChild(div);
    messagesContainer.scrollTop = messages.scrollHeight;
}

function showNotification(message, type = "info") {
    addMessage(message, "bot");
}

function updateButtonState(loading) {
    const sendButton = getElement('send-btn');
    const questionInput = getElement('question');
    
    if (sendButton) {
        sendButton.disabled = loading;
        sendButton.innerHTML = loading ? 
            'Processing...' : 
            '<span>Send</span><span>→</span>';
    }
    
    if (questionInput) questionInput.disabled = loading;
    chatState.isProcessing = loading;
}

// ==== API Functions ====
async function handleApiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Accept': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw new Error('Cannot connect to server. Please make sure the backend is running.');
    }
}

async function checkBackendHealth() {
    try {
        const response = await handleApiCall(`${API_BASE}/health`);
        chatState.isConnected = true;
        chatState.backendHealth = 'healthy';
        showNotification("✅ Connected to backend server", "success");
    } catch (error) {
        chatState.isConnected = false;
        chatState.backendHealth = 'disconnected';
        showNotification("❌ Cannot connect to backend server", "error");
    }
}

async function uploadFiles(fileList) {
    if (!fileList.length) return;
    
    updateButtonState(true);
    showNotification(`Uploading ${fileList.length} file(s)...`, "info");

    try {
        const formData = new FormData();
        Array.from(fileList).forEach(file => formData.append("files", file));

        const response = await handleApiCall(`${API_BASE}/upload`, {
            method: "POST",
            body: formData
        });

        showNotification(`✅ Uploaded ${fileList.length} file(s) successfully`, "success");
        chatState.uploadedFiles = [...chatState.uploadedFiles, ...Array.from(fileList)];
    } catch (error) {
        showNotification(`❌ Upload failed: ${error.message}`, "error");
    } finally {
        updateButtonState(false);
    }
}

async function askQuestion(question) {
    if (!question.trim() || chatState.isProcessing) return;
    
    updateButtonState(true);
    addMessage(question, "user");

    try {
        const response = await handleApiCall(`${API_BASE}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question })
        });

        addMessage(response.answer || "I couldn't find an answer.", "bot");
    } catch (error) {
        showNotification(`❌ Error: ${error.message}`, "error");
    } finally {
        updateButtonState(false);
    }
}

// ==== Event Handlers ====
function handleAskFormSubmit(e) {
    e.preventDefault();
    
    const questionInput = getElement('question');
    if (!questionInput) return;
    
    const question = questionInput.value.trim();
    if (!question) return;
    
    questionInput.value = "";
    askQuestion(question);
}

function handleFileInputChange(e) {
    if (e.target.files.length > 0) {
        uploadFiles(e.target.files);
        e.target.value = "";
    }
}

// ==== Start the application ====
document.addEventListener('DOMContentLoaded', function() {
    // Add minimal styles
    const style = document.createElement('style');
    style.textContent = `
        .status-indicator { margin-right: 8px; }
        .status-connected { color: green; }
        .status-error { color: red; }
        #send-btn:disabled { opacity: 0.6; cursor: not-allowed; }
    `;
    document.head.appendChild(style);

    // Initialize chatbot
    setTimeout(initializeChatbot, 100);
});

// Debug helper
window.debugChatbot = {
    state: chatState,
    checkHealth: checkBackendHealth,
    testQuestion: (q) => askQuestion(q || "Test question")
};