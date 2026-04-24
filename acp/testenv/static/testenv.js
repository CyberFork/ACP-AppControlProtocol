/**
 * ACP 测试环境共享 JS 工具库
 * 提供：Toast、Modal、Tab 切换、表单验证等通用功能
 */

// ── Toast ──────────────────────────────────────────────────────────────────────
const Toast = (() => {
  let container = null;
  function getContainer() {
    if (!container) {
      container = document.createElement('div');
      container.className = 'toast-container';
      document.body.appendChild(container);
    }
    return container;
  }
  function show(message, type = 'info', duration = 3000) {
    const icons = { success: '✓', error: '✕', warning: '⚠', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.setAttribute('data-acp-type', 'toast');
    toast.setAttribute('data-acp-id', `toast-${Date.now()}`);
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    getContainer().appendChild(toast);
    if (duration > 0) setTimeout(() => toast.remove(), duration);
    return toast;
  }
  return { show, success: m => show(m, 'success'), error: m => show(m, 'error'),
           warning: m => show(m, 'warning'), info: m => show(m, 'info') };
})();

// ── Modal ─────────────────────────────────────────────────────────────────────
function openModal(id) {
  const overlay = document.getElementById(id);
  if (overlay) overlay.classList.add('active');
}
function closeModal(id) {
  const overlay = document.getElementById(id);
  if (overlay) overlay.classList.remove('active');
}
// Close modal on overlay click
document.addEventListener('click', e => {
  if (e.target.classList.contains('modal-overlay')) e.target.classList.remove('active');
  if (e.target.classList.contains('modal-close')) {
    e.target.closest('.modal-overlay')?.classList.remove('active');
  }
});

// ── Tabs ──────────────────────────────────────────────────────────────────────
function initTabs(containerEl) {
  const btns = containerEl.querySelectorAll('.tab-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      containerEl.closest('.tabs-wrapper')?.querySelectorAll('.tab-content')
        .forEach(c => c.classList.toggle('active', c.id === target));
    });
  });
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tabs').forEach(initTabs);
});

// ── Form Validation ───────────────────────────────────────────────────────────
function validateForm(formEl) {
  let valid = true;
  formEl.querySelectorAll('[data-required]').forEach(input => {
    const group = input.closest('.form-group');
    const errEl = group?.querySelector('.form-error');
    if (!input.value.trim()) {
      input.classList.add('error');
      if (errEl) errEl.textContent = '此字段为必填项';
      valid = false;
    } else {
      input.classList.remove('error');
      if (errEl) errEl.textContent = '';
    }
  });
  formEl.querySelectorAll('[data-type="email"]').forEach(input => {
    if (input.value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input.value)) {
      input.classList.add('error');
      const group = input.closest('.form-group');
      const errEl = group?.querySelector('.form-error');
      if (errEl) errEl.textContent = '请输入有效的邮箱地址';
      valid = false;
    }
  });
  return valid;
}

// ── Autocomplete ──────────────────────────────────────────────────────────────
function initAutocomplete(inputEl, suggestions) {
  const wrapper = inputEl.parentElement;
  let dropdown = wrapper.querySelector('.autocomplete-dropdown');
  if (!dropdown) {
    dropdown = document.createElement('div');
    dropdown.className = 'autocomplete-dropdown';
    dropdown.style.cssText = `
      position:absolute; top:100%; left:0; right:0; background:white;
      border:1px solid var(--border); border-top:none; border-radius:0 0 8px 8px;
      box-shadow:var(--shadow); z-index:100; max-height:200px; overflow-y:auto;
    `;
    wrapper.style.position = 'relative';
    wrapper.appendChild(dropdown);
  }
  inputEl.addEventListener('input', () => {
    const val = inputEl.value.toLowerCase();
    if (!val) { dropdown.innerHTML = ''; return; }
    const matches = suggestions.filter(s => s.toLowerCase().includes(val)).slice(0, 6);
    dropdown.innerHTML = matches.map(m =>
      `<div class="autocomplete-item" data-acp-type="autocomplete-item" style="padding:8px 12px;cursor:pointer;font-size:14px;"
       onmouseover="this.style.background='#f3f4f6'" onmouseout="this.style.background=''"
       onclick="document.getElementById('${inputEl.id}').value='${m}';this.parentElement.innerHTML=''">${m}</div>`
    ).join('');
  });
  document.addEventListener('click', e => {
    if (!wrapper.contains(e.target)) dropdown.innerHTML = '';
  });
}

// ── Notification Banner ────────────────────────────────────────────────────────
function dismissBanner(el) {
  el.style.animation = 'slideOut 0.2s ease forwards';
  setTimeout(() => el.remove(), 200);
}

// ── Context Menu ──────────────────────────────────────────────────────────────
let activeContextMenu = null;
function showContextMenu(x, y, items) {
  hideContextMenu();
  const menu = document.createElement('div');
  menu.className = 'context-menu';
  menu.setAttribute('data-acp-type', 'context-menu');
  items.forEach(item => {
    if (item === 'sep') {
      menu.innerHTML += '<div class="context-menu-sep"></div>';
    } else {
      const el = document.createElement('div');
      el.className = `context-menu-item${item.danger ? ' danger' : ''}`;
      el.innerHTML = `${item.icon ? item.icon + ' ' : ''}${item.label}`;
      el.onclick = () => { item.action?.(); hideContextMenu(); };
      menu.appendChild(el);
    }
  });
  menu.style.left = x + 'px';
  menu.style.top = y + 'px';
  document.body.appendChild(menu);
  activeContextMenu = menu;
  // Adjust if off-screen
  const rect = menu.getBoundingClientRect();
  if (rect.right > window.innerWidth) menu.style.left = (x - rect.width) + 'px';
  if (rect.bottom > window.innerHeight) menu.style.top = (y - rect.height) + 'px';
}
function hideContextMenu() {
  activeContextMenu?.remove();
  activeContextMenu = null;
}
document.addEventListener('click', hideContextMenu);
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideContextMenu(); });

// ── Long Press ────────────────────────────────────────────────────────────────
function addLongPress(el, callback, duration = 600) {
  let timer;
  el.addEventListener('mousedown', e => {
    timer = setTimeout(() => callback(e), duration);
  });
  el.addEventListener('mouseup', () => clearTimeout(timer));
  el.addEventListener('mouseleave', () => clearTimeout(timer));
  el.addEventListener('touchstart', e => {
    timer = setTimeout(() => callback(e.touches[0]), duration);
  }, { passive: true });
  el.addEventListener('touchend', () => clearTimeout(timer));
}

// ── Drag Sort ─────────────────────────────────────────────────────────────────
function initDragSort(listEl) {
  let dragging = null;
  listEl.querySelectorAll('[draggable]').forEach(item => {
    item.addEventListener('dragstart', e => {
      dragging = item;
      item.style.opacity = '0.5';
    });
    item.addEventListener('dragend', () => {
      dragging = null;
      item.style.opacity = '';
    });
    item.addEventListener('dragover', e => {
      e.preventDefault();
      if (dragging && dragging !== item) {
        const rect = item.getBoundingClientRect();
        const midY = rect.top + rect.height / 2;
        if (e.clientY < midY) listEl.insertBefore(dragging, item);
        else listEl.insertBefore(dragging, item.nextSibling);
      }
    });
  });
}

// ── ACP Indicator ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const indicator = document.createElement('div');
  indicator.className = 'acp-indicator';
  indicator.textContent = 'ACP TestEnv';
  document.body.appendChild(indicator);
});
