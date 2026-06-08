/* Jarvis UX — Application Logic */
(function() {
  'use strict';

  const API_BASE = '';

  const state = {
    activeTab: 'dashboard',
    tasks: [],
    notes: [],
    reminders: [],
    memories: [],
    searchResults: [],
    isRecording: false,
    recognition: null,
    theme: localStorage.getItem('jarvis-theme') || 'dark',
  };

  // ── DOM refs ──
  const $ = id => document.getElementById(id);
  const commandInput = $('command-input');
  const micBtn = $('mic-btn');
  const sendBtn = $('send-btn');
  const voiceIndicator = $('voice-indicator');
  const content = $('content');
  const modalOverlay = $('modal-overlay');
  const sidebar = $('sidebar');

  // ── API Client ──
  const api = {
    async _fetch(method, path, body) {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(API_BASE + path, opts);
      if (method === 'DELETE' && res.status === 204) return null;
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || err.error || 'Request failed');
      }
      return res.status === 204 ? null : res.json();
    },
    tasks: {
      list: (params = {}) => {
        const q = new URLSearchParams(params).toString();
        return api._fetch('GET', `/api/tasks${q ? '?' + q : ''}`);
      },
      create: data => api._fetch('POST', '/api/tasks', data),
      get: id => api._fetch('GET', `/api/tasks/${id}`),
      update: (id, data) => api._fetch('PATCH', `/api/tasks/${id}`, data),
      complete: id => api._fetch('PATCH', `/api/tasks/${id}/complete`),
      delete: id => api._fetch('DELETE', `/api/tasks/${id}`),
      search: q => api._fetch('GET', `/api/tasks/search?q=${encodeURIComponent(q)}`),
    },
    notes: {
      list: (params = {}) => {
        const q = new URLSearchParams(params).toString();
        return api._fetch('GET', `/api/notes${q ? '?' + q : ''}`);
      },
      create: data => api._fetch('POST', '/api/notes', data),
      get: id => api._fetch('GET', `/api/notes/${id}`),
      update: (id, data) => api._fetch('PATCH', `/api/notes/${id}`, data),
      delete: id => api._fetch('DELETE', `/api/notes/${id}`),
    },
    reminders: {
      list: () => api._fetch('GET', '/api/reminders'),
      create: data => api._fetch('POST', '/api/reminders', data),
      delete: id => api._fetch('DELETE', `/api/reminders/${id}`),
    },
    memory: {
      list: () => api._fetch('GET', '/api/memory'),
      store: data => api._fetch('POST', '/api/memory', data),
      delete: id => api._fetch('DELETE', `/api/memory/${id}`),
    },
    voice: {
      command: text => api._fetch('POST', '/api/voice/command', { text }),
    },
    search: {
      all: (q, type) => {
        const params = new URLSearchParams({ q });
        if (type) params.set('type', type);
        return api._fetch('GET', `/api/search?${params}`);
      },
    },
  };

  // ── Theme ──
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    $('theme-toggle').textContent = theme === 'dark' ? '🌙' : '☀️';
    localStorage.setItem('jarvis-theme', theme);
    state.theme = theme;
  }
  applyTheme(state.theme);
  $('theme-toggle').onclick = () => applyTheme(state.theme === 'dark' ? 'light' : 'dark');

  // ── Tab Navigation ──
  function switchTab(tab) {
    state.activeTab = tab;
    document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
    const tabEl = document.getElementById('tab-' + tab);
    if (tabEl) tabEl.classList.add('active');
    const navEl = document.querySelector(`.nav-item[data-tab="${tab}"]`);
    if (navEl) navEl.classList.add('active');
    const titles = { dashboard: 'Dashboard', tasks: 'Tasks', notes: 'Notes', reminders: 'Reminders', memory: 'Memory', search: 'Search' };
    $('page-title').textContent = titles[tab] || tab;
    closeSidebar();
    if (tab === 'tasks') loadTasks();
    else if (tab === 'notes') loadNotes();
    else if (tab === 'reminders') loadReminders();
    else if (tab === 'memory') loadMemory();
    else if (tab === 'dashboard') loadDashboard();
    else if (tab === 'search') $('search-input').focus();
  }

  document.querySelectorAll('.nav-item[data-tab]').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      switchTab(el.dataset.tab);
    });
  });

  // Hash-based routing
  window.addEventListener('hashchange', () => {
    const tab = location.hash.replace('#', '') || 'dashboard';
    if (document.querySelector(`.nav-item[data-tab="${tab}"]`)) switchTab(tab);
  });
  const initialTab = location.hash.replace('#', '') || 'dashboard';
  if (document.querySelector(`.nav-item[data-tab="${initialTab}"]`)) switchTab(initialTab);

  // Menu toggle for mobile
  $('menu-toggle').onclick = () => sidebar.classList.toggle('open');
  function closeSidebar() { if (window.innerWidth <= 768) sidebar.classList.remove('open'); }

  // ── Modal ──
  let modalResolve = null;
  function openModal(title, bodyHtml, footerHtml) {
    $('modal-title').textContent = title;
    $('modal-body').innerHTML = bodyHtml;
    $('modal-footer').innerHTML = footerHtml || '';
    modalOverlay.classList.add('open');
    return new Promise(resolve => { modalResolve = resolve; });
  }
  function closeModal(result) {
    modalOverlay.classList.remove('open');
    if (modalResolve) { modalResolve(result); modalResolve = null; }
  }
  $('modal-close').onclick = () => closeModal(null);
  modalOverlay.onclick = e => { if (e.target === modalOverlay) closeModal(null); };

  // ── Voice Recognition ──
  function initVoice() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      micBtn.style.display = 'none';
      return;
    }
    state.recognition = new SR();
    state.recognition.continuous = false;
    state.recognition.interimResults = false;
    state.recognition.lang = 'en-US';
    state.recognition.onresult = e => {
      const text = e.results[0][0].transcript;
      commandInput.value = text;
      stopRecording();
      processCommand(text);
    };
    state.recognition.onerror = () => stopRecording();
    state.recognition.onend = () => stopRecording();
  }

  function toggleRecording() {
    if (state.isRecording) { stopRecording(); return; }
    if (!state.recognition) { alert('Speech recognition not available in this browser. Try Chrome.'); return; }
    try {
      state.recognition.start();
      state.isRecording = true;
      micBtn.classList.add('recording');
      voiceIndicator.classList.add('listening');
      commandInput.placeholder = 'Listening...';
    } catch (e) {
      console.error('Voice start error:', e);
    }
  }

  function stopRecording() {
    state.isRecording = false;
    micBtn.classList.remove('recording');
    voiceIndicator.classList.remove('listening');
    commandInput.placeholder = 'Type a command or press the mic to speak...';
    try { state.recognition?.stop(); } catch (e) {}
  }

  micBtn.onclick = toggleRecording;

  // ── Process Commands ──
  async function processCommand(text) {
    if (!text.trim()) return;
    // Show activity feedback
    showAlert('info', `Processing: "${text}"`);
    commandInput.value = '';

    try {
      const result = await api.voice.command(text);
      showAlert(result.intent === 'UNKNOWN' ? 'error' : 'success', result.spoken_response);
      // Refresh current tab data
      if (state.activeTab === 'dashboard') loadDashboard();
      else if (state.activeTab === 'tasks' || result.intent.startsWith('TASK_')) loadTasks();
      else if (state.activeTab === 'notes' || result.intent.startsWith('NOTE_')) loadNotes();
      else if (result.intent.startsWith('MEMORY_')) loadMemory();
      else if (result.intent.startsWith('REMINDER_')) loadReminders();
    } catch (err) {
      showAlert('error', `Error: ${err.message}`);
    }
  }

  sendBtn.onclick = () => processCommand(commandInput.value);
  commandInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') processCommand(commandInput.value);
  });

  // ── Dashboard ──
  async function loadDashboard() {
    try {
      const [tasks, notes, mem] = await Promise.all([
        api.tasks.list(),
        api.notes.list({ limit: 3 }),
        api.memory.list(),
      ]);
      const pending = tasks.tasks.filter(t => t.status !== 'completed').length;
      $('stat-tasks').textContent = tasks.total;
      $('stat-pending').textContent = pending;
      $('stat-notes').textContent = notes.total;
      $('stat-memories').textContent = mem.total;
      $('task-badge').textContent = pending || '';
      $('note-badge').textContent = notes.total || '';
      $('reminder-badge').textContent = '';

      // Recent activity
      let html = '';
      const recent = [];
      tasks.tasks.slice(0, 3).forEach(t => recent.push(`<div class="list-item"><span class="item-icon">☑</span><div class="item-body"><div class="item-title">${esc(t.title)}</div><div class="item-meta">task · ${t.status}</div></div></div>`));
      notes.notes.slice(0, 3).forEach(n => recent.push(`<div class="list-item"><span class="item-icon">📄</span><div class="item-body"><div class="item-title">${esc(n.title)}</div><div class="item-meta">note</div></div></div>`));
      mem.entries.slice(0, 3).forEach(m => recent.push(`<div class="list-item"><span class="item-icon">🧠</span><div class="item-body"><div class="item-title">${esc(m.fact)}</div><div class="item-meta">memory · importance ${m.importance}</div></div></div>`));
      $('recent-activity').innerHTML = recent.length ? recent.join('') : '<p class="muted">No recent activity.</p>';
      $('reminder-badge').textContent = '';
    } catch (err) { console.error('Dashboard load error:', err); }
  }

  // ── Tasks ──
  async function loadTasks() {
    const status = $('task-filter-status').value;
    try {
      const result = status ? await api.tasks.list({ status }) : await api.tasks.list();
      state.tasks = result.tasks;
      renderTasks();
    } catch (err) { showAlert('error', `Failed to load tasks: ${err.message}`); }
  }

  function renderTasks() {
    const list = $('task-list');
    if (!state.tasks.length) {
      list.innerHTML = '<div class="empty-state"><div class="icon">☑</div><p>No tasks yet. Create one with the button above or say "create task..."</p></div>';
      return;
    }
    list.innerHTML = state.tasks.map(t => `
      <div class="list-item">
        <span class="item-icon">${t.status === 'completed' ? '✅' : '◻'}</span>
        <div class="item-body">
          <div class="item-title" style="${t.status === 'completed' ? 'text-decoration:line-through;color:var(--text-muted)' : ''}">${esc(t.title)}</div>
          <div class="item-meta">
            ${t.description ? esc(t.description.substring(0,80)) : ''}
            ${t.priority ? `<span class="tag priority-${t.priority}">${t.priority}</span>` : ''}
            ${t.status ? `<span class="tag status-${t.status}">${t.status}</span>` : ''}
            ${t.due_date ? ` · due ${new Date(t.due_date).toLocaleDateString()}` : ''}
          </div>
        </div>
        <div class="item-actions">
          ${t.status !== 'completed' ? `<button class="complete-btn" data-id="${t.id}" title="Complete">✓</button>` : ''}
          <button class="delete-btn" data-id="${t.id}" data-type="task" title="Delete">✕</button>
        </div>
      </div>
    `).join('');
  }

  $('task-filter-status').onchange = loadTasks;
  $('task-create-btn').onclick = () => showTaskForm();

  content.addEventListener('click', e => {
    const btn = e.target.closest('.complete-btn');
    if (btn) {
      api.tasks.complete(btn.dataset.id).then(() => loadTasks()).catch(err => showAlert('error', err.message));
      return;
    }
    const del = e.target.closest('.delete-btn');
    if (del && del.dataset.type === 'task') {
      if (confirm('Delete this task?'))
        api.tasks.delete(del.dataset.id).then(() => loadTasks()).catch(err => showAlert('error', err.message));
    }
  });

  function showTaskForm(data) {
    const isEdit = !!data;
    openModal(isEdit ? 'Edit Task' : 'New Task', `
      <div class="form-group"><label>Title</label><input id="f-title" value="${data ? esc(data.title) : ''}" /></div>
      <div class="form-group"><label>Description</label><textarea id="f-desc">${data ? esc(data.description || '') : ''}</textarea></div>
      <div class="form-row">
        <div class="form-group"><label>Priority</label><select id="f-priority"><option ${data && data.priority === 'low' ? 'selected' : ''}>low</option><option ${!data || data.priority === 'medium' ? 'selected' : ''}>medium</option><option ${data && data.priority === 'high' ? 'selected' : ''}>high</option></select></div>
        <div class="form-group"><label>Status</label><select id="f-status"><option ${!data || data.status === 'pending' ? 'selected' : ''}>pending</option><option ${data && data.status === 'completed' ? 'selected' : ''}>completed</option></select></div>
      </div>
    `, `<button class="btn" onclick="closeModal(null)">Cancel</button><button class="btn btn-primary" id="f-save">${isEdit ? 'Update' : 'Create'}</button>`);

    $('f-save').onclick = async () => {
      const body = { title: $('f-title').value, description: $('f-desc').value, priority: $('f-priority').value };
      if (isEdit) body.status = $('f-status').value;
      try {
        isEdit ? await api.tasks.update(data.id, body) : await api.tasks.create(body);
        closeModal(true);
        loadTasks();
      } catch (err) { showAlert('error', err.message); }
    };
  }

  // ── Notes ──
  async function loadNotes() {
    try {
      const result = await api.notes.list();
      state.notes = result.notes;
      renderNotes();
    } catch (err) { showAlert('error', `Failed to load notes: ${err.message}`); }
  }

  function renderNotes() {
    const list = $('note-list');
    if (!state.notes.length) {
      list.innerHTML = '<div class="empty-state"><div class="icon">📄</div><p>No notes yet. Say "create note about..."</p></div>';
      return;
    }
    list.innerHTML = state.notes.map(n => `
      <div class="list-item">
        <span class="item-icon">📄</span>
        <div class="item-body">
          <div class="item-title">${esc(n.title)}</div>
          <div class="item-meta">${n.content ? esc(n.content.substring(0, 100)) : ''} ${n.tags ? '· ' + n.tags : ''}</div>
        </div>
        <div class="item-actions">
          <button class="edit-note-btn" data-id="${n.id}" title="Edit">✎</button>
          <button class="delete-btn" data-id="${n.id}" data-type="note" title="Delete">✕</button>
        </div>
      </div>
    `).join('');
  }

  $('note-create-btn').onclick = () => showNoteForm();

  content.addEventListener('click', e => {
    const edit = e.target.closest('.edit-note-btn');
    if (edit) {
      const note = state.notes.find(n => n.id == edit.dataset.id);
      if (note) showNoteForm(note);
      return;
    }
    const del = e.target.closest('.delete-btn[data-type="note"]');
    if (del) {
      if (confirm('Delete this note?'))
        api.notes.delete(del.dataset.id).then(() => loadNotes()).catch(err => showAlert('error', err.message));
    }
  });

  function showNoteForm(data) {
    openModal(data ? 'Edit Note' : 'New Note', `
      <div class="form-group"><label>Title</label><input id="nf-title" value="${data ? esc(data.title) : ''}" /></div>
      <div class="form-group"><label>Content</label><textarea id="nf-content">${data ? esc(data.content || '') : ''}</textarea></div>
    `, `<button class="btn" onclick="closeModal(null)">Cancel</button><button class="btn btn-primary" id="nf-save">${data ? 'Update' : 'Create'}</button>`);
    $('nf-save').onclick = async () => {
      const body = { title: $('nf-title').value, content: $('nf-content').value };
      try {
        data ? await api.notes.update(data.id, body) : await api.notes.create(body);
        closeModal(true);
        loadNotes();
      } catch (err) { showAlert('error', err.message); }
    };
  }

  // ── Reminders ──
  async function loadReminders() {
    try {
      const result = await api.reminders.list();
      state.reminders = result.reminders;
      renderReminders();
    } catch (err) { showAlert('error', `Failed to load reminders: ${err.message}`); }
  }

  function renderReminders() {
    const list = $('reminder-list');
    if (!state.reminders.length) {
      list.innerHTML = '<div class="empty-state"><div class="icon">⏰</div><p>No reminders. Say "remind me to..."</p></div>';
      return;
    }
    list.innerHTML = state.reminders.map(r => `
      <div class="list-item">
        <span class="item-icon">⏰</span>
        <div class="item-body">
          <div class="item-title">${esc(r.title)}</div>
          <div class="item-meta">${new Date(r.reminder_time).toLocaleString()} · ${r.status} ${r.repeat_type !== 'none' ? '· repeats: ' + r.repeat_type : ''}</div>
        </div>
        <div class="item-actions">
          <button class="delete-btn" data-id="${r.id}" data-type="reminder" title="Delete">✕</button>
        </div>
      </div>
    `).join('');
  }

  $('reminder-create-btn').onclick = () => showReminderForm();

  content.addEventListener('click', e => {
    const del = e.target.closest('.delete-btn[data-type="reminder"]');
    if (del) {
      if (confirm('Delete this reminder?'))
        api.reminders.delete(del.dataset.id).then(() => loadReminders()).catch(err => showAlert('error', err.message));
    }
  });

  function showReminderForm() {
    const now = new Date();
    const defaultTime = new Date(now.getTime() + 3600000).toISOString().slice(0, 16);
    openModal('New Reminder', `
      <div class="form-group"><label>What</label><input id="rf-title" /></div>
      <div class="form-group"><label>When</label><input type="datetime-local" id="rf-time" value="${defaultTime}" /></div>
    `, `<button class="btn" onclick="closeModal(null)">Cancel</button><button class="btn btn-primary" id="rf-save">Create</button>`);
    $('rf-save').onclick = async () => {
      try {
        await api.reminders.create({ title: $('rf-title').value, reminder_time: $('rf-time').value });
        closeModal(true);
        loadReminders();
      } catch (err) { showAlert('error', err.message); }
    };
  }

  // ── Memory ──
  async function loadMemory() {
    try {
      const result = await api.memory.list();
      state.memories = result.entries;
      renderMemory();
    } catch (err) { showAlert('error', `Failed to load memories: ${err.message}`); }
  }

  function renderMemory() {
    const list = $('memory-list');
    if (!state.memories.length) {
      list.innerHTML = '<div class="empty-state"><div class="icon">🧠</div><p>No memories yet. Say "remember that..." to teach me!</p></div>';
      return;
    }
    list.innerHTML = state.memories.map(m => `
      <div class="list-item">
        <span class="item-icon">🧠</span>
        <div class="item-body">
          <div class="item-title">${esc(m.fact)}</div>
          <div class="item-meta">${m.category} · importance ${m.importance} · ${new Date(m.created_at).toLocaleDateString()}</div>
        </div>
        <div class="item-actions">
          <button class="delete-btn" data-id="${m.id}" data-type="memory" title="Delete">✕</button>
        </div>
      </div>
    `).join('');
  }

  content.addEventListener('click', e => {
    const del = e.target.closest('.delete-btn[data-type="memory"]');
    if (del) {
      if (confirm('Forget this memory?'))
        api.memory.delete(del.dataset.id).then(() => loadMemory()).catch(err => showAlert('error', err.message));
    }
  });

  // ── Search ──
  $('search-btn').onclick = () => doSearch();
  $('search-input').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

  async function doSearch() {
    const q = $('search-input').value.trim();
    if (!q) return;
    try {
      const result = await api.search.all(q);
      state.searchResults = result.results;
      const list = $('search-results');
      if (!result.results.length) {
        list.innerHTML = '<div class="empty-state"><p>No results found.</p></div>';
        return;
      }
      list.innerHTML = result.results.map(r => `
        <div class="list-item">
          <span class="item-icon">${r.type === 'task' ? '☑' : r.type === 'note' ? '📄' : '🧠'}</span>
          <div class="item-body">
            <div class="item-title">${esc(r.title)}</div>
            <div class="item-meta">${r.type} · ${esc(r.snippet)}</div>
          </div>
        </div>
      `).join('');
    } catch (err) { showAlert('error', err.message); }
  }

  // ── Alert/Toast ──
  function showAlert(type, message) {
    const className = type === 'success' ? 'alert-success' : type === 'error' ? 'alert-error' : '';
    const el = document.createElement('div');
    el.className = `alert ${className}`;
    el.textContent = message;
    const cmdBar = document.querySelector('.command-bar');
    cmdBar.parentNode.insertBefore(el, cmdBar);
    setTimeout(() => el.remove(), type === 'error' ? 6000 : 4000);
  }

  function esc(s) {
    const div = document.createElement('div');
    div.textContent = s || '';
    return div.innerHTML;
  }

  // ── Init ──
  initVoice();
  if (state.activeTab === 'dashboard') loadDashboard();
  setInterval(() => {
    if (state.activeTab === 'dashboard') loadDashboard();
  }, 30000);
})();
