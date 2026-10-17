(() => {
  // ---------- Config / persistence ----------
  const LS_CHATS = 'ultron_chats';
  const LS_API_BASE = 'ultron_api_base';
  const LS_EFFORT = 'ultron_effort';
  const DEFAULT_API_BASE = 'http://localhost:8000';

  let apiBase = localStorage.getItem(LS_API_BASE) || DEFAULT_API_BASE;
  let currentEffort = localStorage.getItem(LS_EFFORT) || 'auto';
  let chats = loadChats();
  let activeChatId = chats.length ? chats[0].id : null;
  let isStreaming = false;
  let currentController = null;
  let pendingImages = []; // base64 strings

  const els = {
    sidebar: document.getElementById('sidebar'),
    main: document.querySelector('.main'),
    collapseBtn: document.getElementById('collapseBtn'),
    openSidebarBtn: document.getElementById('openSidebarBtn'),
    newChatBtn: document.getElementById('newChatBtn'),
    chatList: document.getElementById('chatList'),
    searchInput: document.getElementById('searchInput'),
    chatArea: document.getElementById('chatArea'),
    welcome: document.getElementById('welcome'),
    messages: document.getElementById('messages'),
    composerForm: document.getElementById('composerForm'),
    composerInput: document.getElementById('composerInput'),
    sendBtn: document.getElementById('sendBtn'),
    statusBadge: document.getElementById('statusBadge'),
    statusDot: document.getElementById('statusDot'),
    settingsMenu: document.getElementById('settingsMenu'),
    apiBaseInput: document.getElementById('apiBaseInput'),
    settingsStatus: document.getElementById('settingsStatus'),
    testConnBtn: document.getElementById('testConnBtn'),
    apiHint: document.getElementById('apiHint'),
    suggestionRow: document.getElementById('suggestionRow'),
    effortSelector: document.getElementById('effortSelector'),
    fileInput: document.getElementById('fileInput'),
    imagePreview: document.getElementById('imagePreview'),
    attachBtn: document.querySelector('.attach-btn'),
  };

  const uid = () => Math.random().toString(36).slice(2, 10);

  // ---------- Persistence helpers ----------
  function loadChats() {
    try {
      const raw = localStorage.getItem(LS_CHATS);
      const parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  function saveChats() {
    try { localStorage.setItem(LS_CHATS, JSON.stringify(chats)); } catch {}
  }

  // ---------- Chat management ----------
  function createChat() {
    const chat = { id: uid(), title: 'New chat', sessionId: null, messages: [] };
    chats.unshift(chat);
    activeChatId = chat.id;
    saveChats();
    renderSidebar();
    renderMessages();
    els.composerInput.value = '';
    autoResize();
    els.composerInput.focus();
    return chat;
  }

  function getActiveChat() {
    return chats.find(c => c.id === activeChatId) || null;
  }

  function switchChat(id) {
    activeChatId = id;
    renderSidebar();
    renderMessages();
    if (window.innerWidth <= 768) closeSidebar();
  }

  function deleteChat(id, evt) {
    evt.stopPropagation();
    const chat = chats.find(c => c.id === id);
    if (chat && chat.sessionId) {
      fetch(`${apiBase}/session/${chat.sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
    chats = chats.filter(c => c.id !== id);
    if (activeChatId === id) {
      activeChatId = chats.length ? chats[0].id : null;
    }
    saveChats();
    renderSidebar();
    renderMessages();
  }

  function titleFromText(text) {
    const clean = text.trim().replace(/\s+/g, ' ');
    return clean.length > 42 ? clean.slice(0, 42) + '…' : (clean || 'New chat');
  }

  // ---------- Rendering ----------
  function renderSidebar() {
    const q = els.searchInput.value.trim().toLowerCase();
    els.chatList.innerHTML = '';
    const filtered = chats.filter(c => c.title.toLowerCase().includes(q));

    if (!filtered.length) {
      const empty = document.createElement('div');
      empty.className = 'chat-list-label';
      empty.textContent = q ? 'No matches' : 'No chats yet';
      els.chatList.appendChild(empty);
      return;
    }

    const label = document.createElement('div');
    label.className = 'chat-list-label';
    label.textContent = 'Chats';
    els.chatList.appendChild(label);

    filtered.forEach(chat => {
      const item = document.createElement('div');
      item.className = 'chat-item' + (chat.id === activeChatId ? ' active' : '');
      item.innerHTML = `
        <span>${escapeHtml(chat.title)}</span>
        <button class="del-btn" title="Delete">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/></svg>
        </button>`;
      item.addEventListener('click', () => switchChat(chat.id));
      item.querySelector('.del-btn').addEventListener('click', (e) => deleteChat(chat.id, e));
      els.chatList.appendChild(item);
    });
  }

  function renderMessages() {
    const chat = getActiveChat();
    els.messages.innerHTML = '';

    if (!chat || chat.messages.length === 0) {
      els.welcome.style.display = 'flex';
      els.welcome.style.flexDirection = 'column';
      els.welcome.style.justifyContent = 'center';
      els.messages.classList.remove('active');
      return;
    }

    els.welcome.style.display = 'none';
    els.messages.classList.add('active');

    chat.messages.forEach(m => {
      els.messages.appendChild(buildMessageEl(m.role, m.text, m.isError, m.images, m.sources, m.sourceFailures, m.skills, m.skillGuidance));
    });
    scrollToBottom();
  }

  function buildSourcesEl(sources, failures) {
    const list = (sources || []).filter(s => s && s.id);
    const failed = failures || [];
    if (!list.length && !failed.length) return null;

    const box = document.createElement('div');
    box.className = 'msg-sources';

    if (list.length) {
      const head = document.createElement('div');
      head.className = 'msg-sources-head';
      head.textContent = `Sources (${list.length})`;
      box.appendChild(head);

      const ul = document.createElement('ul');
      list.forEach(s => {
        const li = document.createElement('li');
        li.className = 'msg-source' + (s.kind === 'web' ? ' web' : ' kb');

        const label = document.createElement('span');
        label.className = 'msg-source-label';
        label.textContent = s.id;

        const title = document.createElement('span');
        title.className = 'msg-source-title';
        title.textContent = s.title || '';

        li.appendChild(label);
        li.appendChild(title);

        if (s.kind === 'web') {
          if (s.url) {
            const link = document.createElement('a');
            link.className = 'msg-source-link';
            link.href = s.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = s.domain || s.url;
            li.appendChild(link);
          }
        } else {
          const loc = document.createElement('span');
          loc.className = 'msg-source-meta';
          loc.textContent = s.location || '';
          li.appendChild(loc);
        }

        ul.appendChild(li);
      });
      box.appendChild(ul);
    }

    if (failed.length) {
      const warn = document.createElement('div');
      warn.className = 'msg-sources-failed';
      warn.textContent = failed
        .map(f => `${f.tool}: ${f.error}`)
        .join(' · ');
      box.appendChild(warn);
    }

    return box;
  }

  function buildSkillsEl(skills, guidance) {
    const list = (skills || []).filter(s => s && (s.id || s.name));
    if (!list.length) return null;

    const box = document.createElement('div');
    box.className = 'msg-skills';

    const head = document.createElement('div');
    head.className = 'msg-skills-head';
    head.textContent = `Skills used (${list.length})`;
    box.appendChild(head);

    const chips = document.createElement('div');
    chips.className = 'msg-skill-chips';
    list.forEach(s => {
      const chip = document.createElement('span');
      chip.className = 'msg-skill-chip';
      chip.textContent = s.name || s.id;
      const detail = [s.reason, s.path ? `(${s.path})` : '']
        .filter(Boolean).join(' — ');
      if (detail) chip.title = detail;
      chips.appendChild(chip);
    });
    box.appendChild(chips);

    if (guidance) {
      const g = document.createElement('div');
      g.className = 'msg-skill-guidance';
      g.textContent = guidance;
      box.appendChild(g);
    }

    return box;
  }

  function buildMessageEl(role, text, isError, images, sources, sourceFailures, skills, skillGuidance) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'assistant' ? '✦' : '';

    const bubbleWrap = document.createElement('div');
    bubbleWrap.className = 'msg-bubble-wrap';

    if (images && images.length) {
      const imgRow = document.createElement('div');
      imgRow.className = 'msg-images';
      images.forEach(b64 => {
        const img = document.createElement('img');
        img.src = 'data:image/png;base64,' + b64;
        imgRow.appendChild(img);
      });
      bubbleWrap.appendChild(imgRow);
    }

    const content = document.createElement('div');
    content.className = 'msg-content' + (isError ? ' error-content' : '');
    content.innerHTML = formatContent(text);
    bubbleWrap.appendChild(content);

    const sourcesEl = buildSourcesEl(sources, sourceFailures);
    if (sourcesEl) {
      bubbleWrap.appendChild(sourcesEl);
    }

    const skillsEl = buildSkillsEl(skills, skillGuidance);
    if (skillsEl) {
      bubbleWrap.appendChild(skillsEl);
    }

    if (role === 'assistant' && !isError) {
      const actions = document.createElement('div');
      actions.className = 'msg-actions';
      actions.innerHTML = `
        <button title="Copy" class="copy-btn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>
        </button>
        <button title="Regenerate" class="regen-btn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/></svg>
        </button>`;
      actions.querySelector('.copy-btn').addEventListener('click', () => {
        navigator.clipboard?.writeText(text).catch(() => {});
      });
      actions.querySelector('.regen-btn').addEventListener('click', () => regenerateLast());
      bubbleWrap.appendChild(actions);
    }

    wrap.appendChild(avatar);
    wrap.appendChild(bubbleWrap);
    return wrap;
  }

  function formatContent(text) {
    let safe = escapeHtml(text);
    safe = safe.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
    safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
    safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    safe = safe.split(/\n\n+/).map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');
    return safe;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function scrollToBottom() {
    els.chatArea.scrollTop = els.chatArea.scrollHeight;
  }

  // ---------- API ----------
  async function callChatAPI(message, sessionId, signal) {
    const chat = getActiveChat();
    const history = chat
      ? chat.messages.slice(-10).map(m => ({ role: m.role, text: m.text }))
      : [];
    const res = await fetch(`${apiBase}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: message,
        session_id: sessionId || null,
        effort: currentEffort,
        images: pendingImages.length ? pendingImages : null,
        chat_history: history.length ? history : null,
      }),
      signal,
    });
    if (!res.ok) {
      let detail = '';
      try { detail = (await res.json()).detail || ''; } catch {}
      throw new Error(detail || `Request failed with status ${res.status}`);
    }
    return res.json();
  }

  async function checkConnection(showBusy = true) {
    if (showBusy) setStatus('checking');
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 4000);
      const res = await fetch(`${apiBase}/health`, { signal: controller.signal });
      clearTimeout(t);
      if (res.ok) {
        setStatus('online');
        return true;
      }
      setStatus('offline');
      return false;
    } catch {
      setStatus('offline');
      return false;
    }
  }

  function setStatus(state) {
    els.statusDot.classList.remove('online', 'offline', 'checking');
    els.statusDot.classList.add(state);
    els.settingsStatus.classList.remove('online', 'offline');
    if (state === 'online') {
      els.settingsStatus.textContent = `Connected to ${apiBase}`;
      els.settingsStatus.classList.add('online');
      els.apiHint.textContent = '';
    } else if (state === 'offline') {
      els.settingsStatus.textContent = `Can't reach ${apiBase}`;
      els.settingsStatus.classList.add('offline');
      els.apiHint.textContent = `· Not connected to ${apiBase}`;
      els.apiHint.classList.add('error-text');
    } else {
      els.settingsStatus.textContent = 'Checking…';
    }
  }

  // ---------- Sending ----------
  function sendMessage(text) {
    if (!text.trim() || isStreaming) return;
    let chat = getActiveChat();
    if (!chat) chat = createChat();

    if (chat.messages.length === 0) chat.title = titleFromText(text);

    const userImages = pendingImages.length ? [...pendingImages] : null;
    chat.messages.push({ role: 'user', text, images: userImages });
    saveChats();
    renderSidebar();
    renderMessages();

    els.composerInput.value = '';
    autoResize();

    isStreaming = true;
    setSendingUI(true);

    const typingEl = buildTypingEl();
    els.messages.appendChild(typingEl);
    scrollToBottom();

    currentController = new AbortController();

    callChatAPI(text, chat.sessionId, currentController.signal)
      .then(data => {
        typingEl.remove();
        chat.sessionId = data.session_id;
        clearPendingImages();

        const msgEl = buildMessageEl(
          'assistant',
          '',
          false,
          null,
          data.sources,
          data.retrieval_failures,
          data.skills_used,
          data.skill_execution_guidance
        );
        const contentEl = msgEl.querySelector('.msg-content');
        contentEl.classList.add('cursor-blink');
        els.messages.appendChild(msgEl);
        scrollToBottom();

        chat.messages.push({
          role: 'assistant',
          text: data.response,
          sources: data.sources || [],
          sourceFailures: data.retrieval_failures || [],
          skills: data.skills_used || [],
          skillGuidance: data.skill_execution_guidance || ''
        });
        saveChats();

        streamText(contentEl, data.response, () => {
          contentEl.classList.remove('cursor-blink');
          finishSending();
        });
        setStatus('online');
      })
      .catch(err => {
        typingEl.remove();
        clearPendingImages();
        finishSending();
        if (err.name === 'AbortError') {
          return; // user stopped the request; nothing further to show
        }
        const reason = err.message && !err.message.includes('Failed to fetch')
          ? err.message
          : `Couldn't reach ULTRON at ${apiBase}. Check that the server is running (uvicorn main:app) and that the API base URL in settings is correct.`;
        const errText = `⚠ ${reason}`;
        chat.messages.push({ role: 'assistant', text: errText, isError: true });
        saveChats();
        els.messages.appendChild(buildMessageEl('assistant', errText, true));
        scrollToBottom();
        setStatus('offline');
      });
  }

  function regenerateLast() {
    const chat = getActiveChat();
    if (!chat || isStreaming) return;
    const lastUser = [...chat.messages].reverse().find(m => m.role === 'user');
    if (!lastUser) return;
    // Drop the trailing assistant message locally, then resend the last user text.
    if (chat.messages[chat.messages.length - 1].role === 'assistant') {
      chat.messages.pop();
    }
    saveChats();
    renderMessages();
    sendMessage(lastUser.text);
    // sendMessage pushes a new user message; remove the duplicate we already had.
    chat.messages.splice(chat.messages.length - 2, 1);
    saveChats();
  }

  function finishSending() {
    isStreaming = false;
    currentController = null;
    setSendingUI(false);
    saveChats();
    renderSidebar();
  }

  function setSendingUI(sending) {
    els.sendBtn.classList.toggle('stop', sending);
    els.sendBtn.disabled = sending ? false : (els.composerInput.value.trim().length === 0);
    els.sendBtn.title = sending ? 'Stop' : 'Send';
  }

  function buildTypingEl() {
    const wrap = document.createElement('div');
    wrap.className = 'msg assistant';
    wrap.innerHTML = `
      <div class="msg-avatar">✦</div>
      <div class="msg-bubble-wrap">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>`;
    return wrap;
  }

  function streamText(el, fullText, onDone) {
    const formatted = formatContent(fullText);
    let i = 0;
    const step = () => {
      i += Math.max(1, Math.floor(fullText.length / 120));
      const slice = fullText.slice(0, i);
      el.innerHTML = formatContent(slice);
      scrollToBottom();
      if (i < fullText.length) {
        requestAnimationFrame(() => setTimeout(step, 10));
      } else {
        el.innerHTML = formatted;
        onDone();
      }
    };
    step();
  }

  // ---------- Composer ----------
  function autoResize() {
    els.composerInput.style.height = 'auto';
    els.composerInput.style.height = Math.min(els.composerInput.scrollHeight, 200) + 'px';
  }

  els.composerInput.addEventListener('input', () => {
    autoResize();
    if (!isStreaming) els.sendBtn.disabled = els.composerInput.value.trim().length === 0;
  });
  els.composerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      els.composerForm.requestSubmit();
    }
  });
  els.composerForm.addEventListener('submit', (e) => {
    e.preventDefault();
    if (isStreaming) {
      currentController?.abort();
      return;
    }
    sendMessage(els.composerInput.value);
  });

  // ---------- Image handling ----------
  function clearPendingImages() {
    pendingImages = [];
    els.imagePreview.innerHTML = '';
    els.fileInput.value = '';
  }

  function renderImagePreview() {
    els.imagePreview.innerHTML = '';
    pendingImages.forEach((b64, i) => {
      const wrap = document.createElement('div');
      wrap.className = 'preview-thumb';
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + b64;
      const removeBtn = document.createElement('button');
      removeBtn.className = 'preview-remove';
      removeBtn.innerHTML = '&times;';
      removeBtn.addEventListener('click', () => {
        pendingImages.splice(i, 1);
        renderImagePreview();
      });
      wrap.appendChild(img);
      wrap.appendChild(removeBtn);
      els.imagePreview.appendChild(wrap);
    });
  }

  function readAsBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        // Strip the data URI prefix to get raw base64.
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  els.attachBtn.addEventListener('click', () => {
    els.fileInput.click();
  });

  els.fileInput.addEventListener('change', async () => {
    const file = els.fileInput.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) return;
    try {
      const b64 = await readAsBase64(file);
      pendingImages.push(b64);
      renderImagePreview();
    } catch {}
    els.fileInput.value = '';
  });

  // ---------- Paste image handling ----------
  els.composerInput.addEventListener('paste', async (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith('image/')) {
        e.preventDefault();
        const file = item.getAsFile();
        if (!file) continue;
        try {
          const b64 = await readAsBase64(file);
          pendingImages.push(b64);
          renderImagePreview();
        } catch {}
        return;
      }
    }
  });

  // ---------- Sidebar controls ----------
  function closeSidebar() {
    els.sidebar.classList.add('collapsed');
    els.main.classList.add('sidebar-collapsed');
  }
  function openSidebar() {
    els.sidebar.classList.remove('collapsed');
    els.main.classList.remove('sidebar-collapsed');
  }
  els.collapseBtn.addEventListener('click', closeSidebar);
  els.openSidebarBtn.addEventListener('click', openSidebar);
  els.newChatBtn.addEventListener('click', createChat);
  els.searchInput.addEventListener('input', renderSidebar);

  // ---------- Settings / connection ----------
  els.apiBaseInput.value = apiBase;
  els.statusBadge.addEventListener('click', (e) => {
    e.stopPropagation();
    els.settingsMenu.classList.toggle('open');
    els.statusBadge.classList.toggle('open');
    if (els.settingsMenu.classList.contains('open')) els.apiBaseInput.focus();
  });
  document.addEventListener('click', () => {
    els.settingsMenu.classList.remove('open');
    els.statusBadge.classList.remove('open');
  });
  els.settingsMenu.addEventListener('click', (e) => e.stopPropagation());

  els.testConnBtn.addEventListener('click', async () => {
    const val = els.apiBaseInput.value.trim().replace(/\/+$/, '');
    if (!val) return;
    apiBase = val;
    localStorage.setItem(LS_API_BASE, apiBase);
    await checkConnection();
  });
  els.apiBaseInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); els.testConnBtn.click(); }
  });

  // ---------- Suggestion chips ----------
  els.suggestionRow.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      sendMessage(chip.dataset.prompt);
    });
  });

  // ---------- Effort selector ----------
  function setActiveEffort(level) {
    currentEffort = level;
    localStorage.setItem(LS_EFFORT, level);
    els.effortSelector.querySelectorAll('.effort-pill').forEach(pill => {
      pill.classList.toggle('active', pill.dataset.effort === level);
    });
  }

  els.effortSelector.querySelectorAll('.effort-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      setActiveEffort(pill.dataset.effort);
    });
  });

  // Restore saved effort selection.
  setActiveEffort(currentEffort);

  // ---------- Init ----------
  if (window.innerWidth <= 768) closeSidebar();
  if (!chats.length) createChat();
  renderSidebar();
  renderMessages();
  els.sendBtn.disabled = true;
  checkConnection();
  setInterval(() => checkConnection(false), 30000);
})();