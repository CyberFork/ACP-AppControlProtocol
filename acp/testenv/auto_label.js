/**
 * ACP 自动标注脚本 (auto_label.js)
 * 注入每个测试页面，提供：
 *  1. 元素标注导出（遍历 data-acp-type 元素，导出 bbox + type + text）
 *  2. 操作录制（click/input/scroll 事件）
 *  3. 导出为 ACP 训练格式 JSON
 *  4. 截图 + 标注一键导出
 */
(function() {
  'use strict';

  const ACP = window.ACP = {
    recording: false,
    actions: [],
    elements: [],
    sessionId: `session_${Date.now()}`,
  };

  // ── 1. 元素标注 ────────────────────────────────────────────────────────────
  function getBBoxRelative(el) {
    const rect = el.getBoundingClientRect();
    const scX = window.scrollX, scY = window.scrollY;
    return {
      x: Math.round(rect.left + scX),
      y: Math.round(rect.top + scY),
      w: Math.round(rect.width),
      h: Math.round(rect.height),
    };
  }

  function collectElements() {
    const els = [];
    document.querySelectorAll('[data-acp-type]').forEach(el => {
      if (el.offsetParent === null && !el.classList.contains('modal-overlay')) return; // skip hidden
      const bbox = getBBoxRelative(el);
      if (bbox.w === 0 || bbox.h === 0) return;
      els.push({
        id: el.dataset.acpId || el.id || `el_${els.length}`,
        type: el.dataset.acpType,
        bbox: [bbox.x, bbox.y, bbox.w, bbox.h],
        text: (el.value || el.textContent || el.placeholder || '').trim().slice(0, 100),
        tag: el.tagName.toLowerCase(),
        visible: true,
        attrs: {
          disabled: el.disabled || false,
          checked: el.checked,
          selected: el.tagName === 'SELECT' ? el.options[el.selectedIndex]?.text : undefined,
          href: el.href || undefined,
          placeholder: el.placeholder || undefined,
        }
      });
    });
    return els;
  }

  // ── 2. 操作录制 ────────────────────────────────────────────────────────────
  function getTargetInfo(el) {
    if (!el) return null;
    const bbox = getBBoxRelative(el);
    return {
      id: el.dataset?.acpId || el.id || null,
      type: el.dataset?.acpType || el.tagName.toLowerCase(),
      bbox: [bbox.x, bbox.y, bbox.w, bbox.h],
      text: (el.value || el.textContent || '').trim().slice(0, 50),
    };
  }

  function recordAction(action) {
    if (!ACP.recording) return;
    ACP.actions.push({ ...action, timestamp: Date.now(), url: location.href });
  }

  // Click
  document.addEventListener('click', e => {
    const target = e.target.closest('[data-acp-type]') || e.target;
    recordAction({
      type: 'click',
      coord: [Math.round(e.pageX), Math.round(e.pageY)],
      element: getTargetInfo(target),
      button: e.button,
    });
  }, true);

  // Double Click
  document.addEventListener('dblclick', e => {
    const target = e.target.closest('[data-acp-type]') || e.target;
    recordAction({
      type: 'dblclick',
      coord: [Math.round(e.pageX), Math.round(e.pageY)],
      element: getTargetInfo(target),
    });
  }, true);

  // Right Click
  document.addEventListener('contextmenu', e => {
    const target = e.target.closest('[data-acp-type]') || e.target;
    recordAction({
      type: 'right_click',
      coord: [Math.round(e.pageX), Math.round(e.pageY)],
      element: getTargetInfo(target),
    });
  }, true);

  // Input
  document.addEventListener('input', e => {
    const target = e.target;
    recordAction({
      type: 'input',
      element: getTargetInfo(target),
      value: target.value?.slice(-50),
    });
  }, true);

  // Scroll (throttled)
  let scrollTimer;
  document.addEventListener('scroll', e => {
    clearTimeout(scrollTimer);
    scrollTimer = setTimeout(() => {
      recordAction({
        type: 'scroll',
        scrollX: window.scrollX,
        scrollY: window.scrollY,
        element: getTargetInfo(e.target === document ? document.body : e.target),
      });
    }, 200);
  }, true);

  // Keyboard
  document.addEventListener('keydown', e => {
    if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return;
    const combo = [e.ctrlKey && 'Ctrl', e.shiftKey && 'Shift', e.altKey && 'Alt', e.key]
      .filter(Boolean).join('+');
    recordAction({
      type: 'keydown',
      key: combo,
      element: getTargetInfo(document.activeElement),
    });
  }, true);

  // Drag
  document.addEventListener('dragstart', e => {
    recordAction({
      type: 'dragstart',
      coord: [Math.round(e.pageX), Math.round(e.pageY)],
      element: getTargetInfo(e.target.closest('[data-acp-type]') || e.target),
    });
  }, true);

  document.addEventListener('drop', e => {
    recordAction({
      type: 'drop',
      coord: [Math.round(e.pageX), Math.round(e.pageY)],
      element: getTargetInfo(e.target.closest('[data-acp-type]') || e.target),
    });
  }, true);

  // ── 3. 截图 + 导出 ─────────────────────────────────────────────────────────
  async function captureScreenshot() {
    if (!window.html2canvas) {
      console.warn('[ACP] html2canvas 未加载，跳过截图');
      return null;
    }
    try {
      const canvas = await html2canvas(document.body, { useCORS: true, scale: 1 });
      return canvas.toDataURL('image/png');
    } catch (e) {
      console.error('[ACP] 截图失败', e);
      return null;
    }
  }

  async function exportData(withScreenshot = false) {
    const elements = collectElements();
    const screenshot = withScreenshot ? await captureScreenshot() : null;
    const data = {
      session_id: ACP.sessionId,
      page: {
        url: location.href,
        title: document.title,
        width: window.innerWidth,
        height: window.innerHeight,
        scroll_x: window.scrollX,
        scroll_y: window.scrollY,
        timestamp: new Date().toISOString(),
      },
      screenshot: screenshot,
      elements: elements,
      actions: ACP.actions,
      summary: {
        element_count: elements.length,
        action_count: ACP.actions.length,
        element_types: [...new Set(elements.map(e => e.type))],
        action_types: [...new Set(ACP.actions.map(a => a.type))],
      }
    };
    return data;
  }

  function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ── 4. 控制面板 ────────────────────────────────────────────────────────────
  function createControlPanel() {
    const panel = document.createElement('div');
    panel.id = 'acp-control-panel';
    panel.style.cssText = `
      position: fixed; bottom: 24px; left: 16px; z-index: 9999;
      background: rgba(17,24,39,0.95); color: white; padding: 10px 14px;
      border-radius: 10px; font-size: 12px; font-family: monospace;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3); min-width: 200px;
    `;
    panel.innerHTML = `
      <div style="font-weight:700;margin-bottom:8px;color:#a78bfa;">ACP 标注控制台</div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button id="acp-btn-record" style="padding:4px 8px;border-radius:4px;border:none;
          background:#4f46e5;color:white;cursor:pointer;font-size:11px;">▶ 开始录制</button>
        <button id="acp-btn-export-elements" style="padding:4px 8px;border-radius:4px;border:none;
          background:#059669;color:white;cursor:pointer;font-size:11px;">📋 导出元素</button>
        <button id="acp-btn-export-actions" style="padding:4px 8px;border-radius:4px;border:none;
          background:#d97706;color:white;cursor:pointer;font-size:11px;">📝 导出操作</button>
        <button id="acp-btn-export-all" style="padding:4px 8px;border-radius:4px;border:none;
          background:#7c3aed;color:white;cursor:pointer;font-size:11px;">💾 全量导出</button>
        <button id="acp-btn-highlight" style="padding:4px 8px;border-radius:4px;border:none;
          background:#6b7280;color:white;cursor:pointer;font-size:11px;">🔍 高亮元素</button>
      </div>
      <div id="acp-status" style="margin-top:6px;color:#9ca3af;font-size:11px;">
        就绪 | 元素: <span id="acp-el-count">0</span> | 操作: <span id="acp-act-count">0</span>
      </div>
    `;
    document.body.appendChild(panel);

    const updateStatus = () => {
      document.getElementById('acp-el-count').textContent = collectElements().length;
      document.getElementById('acp-act-count').textContent = ACP.actions.length;
    };
    setInterval(updateStatus, 1000);

    document.getElementById('acp-btn-record').addEventListener('click', () => {
      ACP.recording = !ACP.recording;
      const btn = document.getElementById('acp-btn-record');
      btn.textContent = ACP.recording ? '⏹ 停止录制' : '▶ 开始录制';
      btn.style.background = ACP.recording ? '#ef4444' : '#4f46e5';
      document.getElementById('acp-status').style.color = ACP.recording ? '#34d399' : '#9ca3af';
    });

    document.getElementById('acp-btn-export-elements').addEventListener('click', () => {
      const elements = collectElements();
      downloadJSON({ elements, timestamp: new Date().toISOString(), page: location.href },
        `acp_elements_${Date.now()}.json`);
    });

    document.getElementById('acp-btn-export-actions').addEventListener('click', () => {
      downloadJSON({ actions: ACP.actions, session_id: ACP.sessionId },
        `acp_actions_${Date.now()}.json`);
    });

    document.getElementById('acp-btn-export-all').addEventListener('click', async () => {
      const data = await exportData(false);
      downloadJSON(data, `acp_full_${Date.now()}.json`);
    });

    let highlighting = false;
    let overlays = [];
    document.getElementById('acp-btn-highlight').addEventListener('click', () => {
      highlighting = !highlighting;
      overlays.forEach(o => o.remove());
      overlays = [];
      if (highlighting) {
        const colors = { button:'#4f46e5', input:'#059669', select:'#d97706',
          checkbox:'#7c3aed', radio:'#ec4899', switch:'#0891b2',
          slider:'#0d9488', modal:'#dc2626', tab:'#6366f1', list_item:'#f59e0b' };
        collectElements().forEach(elData => {
          const [x, y, w, h] = elData.bbox;
          const ov = document.createElement('div');
          const color = colors[elData.type] || '#6b7280';
          ov.style.cssText = `
            position:absolute;left:${x}px;top:${y}px;width:${w}px;height:${h}px;
            border:2px solid ${color};background:${color}22;pointer-events:none;
            z-index:8888;box-sizing:border-box;
          `;
          const label = document.createElement('div');
          label.style.cssText = `
            position:absolute;top:-1px;left:-1px;background:${color};color:white;
            font-size:9px;padding:1px 3px;white-space:nowrap;font-family:monospace;
          `;
          label.textContent = elData.type;
          ov.appendChild(label);
          document.body.appendChild(ov);
          overlays.push(ov);
        });
      }
    });
  }

  // ── 初始化 ─────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    createControlPanel();
    // Expose to window for external use
    window.ACPLabel = {
      collect: collectElements,
      export: exportData,
      download: downloadJSON,
      recording: () => ACP.recording,
      actions: () => ACP.actions,
      clear: () => { ACP.actions = []; },
    };
    console.log('[ACP] auto_label.js 已加载，使用 window.ACPLabel 访问标注功能');
  });

  // 快捷键：Ctrl+Shift+E 导出全量
  document.addEventListener('keydown', async e => {
    if (e.ctrlKey && e.shiftKey && e.key === 'E') {
      e.preventDefault();
      const data = await exportData(false);
      downloadJSON(data, `acp_full_${Date.now()}.json`);
    }
    // Ctrl+Shift+R 切换录制
    if (e.ctrlKey && e.shiftKey && e.key === 'R') {
      e.preventDefault();
      document.getElementById('acp-btn-record')?.click();
    }
  });

})();
