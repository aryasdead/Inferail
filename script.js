/**
 * InfeRail AI — Definitive Script
 * Powers: background particle canvas, hero image switcher, sample thumbnail gallery,
 * anomaly injection table simulation, modal dialogs, copy command, and nav scroll.
 */

document.addEventListener('DOMContentLoaded', () => {
  initBgCanvas();
  initHeroImageSwitcher();
  initSampleGallery();
  initDashboardControls();
  initModals();
  initScrollNav();
  initCopyCommand();
  initLiveConf();
});

/* =================================================================
   BACKGROUND PARTICLE CANVAS — floating tech node constellation
   ================================================================= */
function initBgCanvas() {
  const canvas = document.getElementById('bgCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, particles = [], dpr = window.devicePixelRatio || 1;

  function resize() {
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
  }
  resize();
  window.addEventListener('resize', () => { resize(); ctx.scale(dpr, dpr); });

  // Create particles
  for (let i = 0; i < 55; i++) {
    particles.push({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.6 + 0.4,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.15,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw particles
    particles.forEach(p => {
      p.x += p.dx;
      p.y += p.dy;
      if (p.x < 0 || p.x > W) p.dx *= -1;
      if (p.y < 0 || p.y > H) p.dy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 240, 255, ${p.alpha})`;
      ctx.fill();
    });

    // Draw connecting lines between close particles
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 240, 255, ${0.12 * (1 - d / 130)})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  }
  draw();
}

/* =================================================================
   HERO IMAGE SWITCHER — Mode toolbar (Side-by-Side, Annotated, etc.)
   ================================================================= */
function initHeroImageSwitcher() {
  const heroImg = document.getElementById('heroImg');
  const modeButtons = document.querySelectorAll('.mode-btn');
  const dTag1 = document.getElementById('dTag1');
  const dTag2 = document.getElementById('dTag2');

  // Class/severity to display on each image (supports filename or images/ path)
  const imageMeta = {
    'annotated_user_track.jpg':    { t1: 'CRACK DETECTED', d1: 'crack · 89.0% conf · Severity 4', t2: 'JOINT DEFECT DETECTED', d2: 'joint_defect · 78.0% · Severity 3' },
    'side_by_side_user_track.jpg': { t1: 'DUAL INSPECTION', d1: 'Original vs YOLOv8 Annotated', t2: 'DEFECT COUNT', d2: '3 Flaws Flagged' },
    'enhanced.jpg':                { t1: 'CLAHE ENHANCED', d1: 'Local contrast normalized (clip=2.0)', t2: 'BALLAST NOISE', d2: 'Bilateral filtered' },
    'track_sample_crack_shadow.jpg': { t1: 'CRACK + SHADOW', d1: 'crack under low ambient · 92.4%', t2: '', d2: '' },
  };

  modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      modeButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const src = btn.dataset.img;
      const key = src.replace(/^images\//, '');
      if (heroImg) {
        heroImg.style.opacity = '0';
        setTimeout(() => {
          heroImg.src = src;
          heroImg.style.opacity = '1';
        }, 180);
      }

      const meta = imageMeta[key] || imageMeta[src];
      if (meta && dTag1) {
        if (meta.t1) dTag1.querySelector('.dt-title').textContent = meta.t1;
        if (meta.d1) dTag1.querySelector('.dt-body').textContent = meta.d1;
      }
      if (meta && dTag2) {
        if (meta.d2) {
          dTag2.style.display = 'flex';
          if (meta.t2) dTag2.querySelector('.dt-title').textContent = meta.t2;
          dTag2.querySelector('.dt-body').textContent = meta.d2;
        } else {
          dTag2.style.display = 'none';
        }
      }
    });
  });
}

/* =================================================================
   SAMPLE IMAGE GALLERY — Thumbnail click updates big preview
   ================================================================= */
function initSampleGallery() {
  const thumbCards = document.querySelectorAll('.thumb-card');
  const bigImg     = document.getElementById('sampleBigImg');
  const infoTitle  = document.getElementById('sampleTitle');
  const infoDesc   = document.getElementById('sampleDesc');
  const infoClass  = document.getElementById('sampleClass');

  thumbCards.forEach(card => {
    card.addEventListener('click', () => {
      // Update active class
      thumbCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');

      const img   = card.dataset.img;
      const title = card.dataset.title;
      const desc  = card.dataset.desc;
      const cls   = card.dataset.class;

      if (bigImg) {
        bigImg.style.opacity = '0';
        setTimeout(() => {
          bigImg.src = img;
          bigImg.alt = title;
          bigImg.style.opacity = '1';
        }, 180);
      }
      if (infoTitle)  infoTitle.textContent  = title;
      if (infoDesc)   infoDesc.textContent   = desc;
      if (infoClass)  infoClass.textContent  = cls;
    });
  });

  // Smooth opacity transition
  if (bigImg) bigImg.style.transition = 'opacity 0.2s ease';
}

/* =================================================================
   DASHBOARD CONTROLS — Inject anomaly into the defect table
   ================================================================= */
function initDashboardControls() {
  const btnSimulate = document.getElementById('btnSimulate');
  const defectTbody = document.getElementById('defectTbody');
  const critActiveNum = document.getElementById('critActiveNum');
  const defectCount   = document.getElementById('defectCount');
  const btnExportCSV  = document.getElementById('btnExportCSV');

  // Real InfeRail defect class colors from visualizer.py CLASS_COLORS
  const CLASS_CONFIG = {
    crack:            { color: '#FF2A4D', tier: 'HIGH',     badge: 'badge-high' },
    missing_fastener: { color: '#FFAA00', tier: 'MEDIUM',   badge: 'badge-med'  },
    broken_rail:      { color: '#D600FF', tier: 'CRITICAL', badge: 'badge-crit' },
    joint_defect:     { color: '#00D4FF', tier: 'MONITOR',  badge: 'badge-low'  },
  };

  const classNames = Object.keys(CLASS_CONFIG);
  let idCounter = 4;
  let critCount = 0;
  let totalCount = 3;

  const REASONS = {
    crack: 'Surface crack exceeds severity threshold — rapid dispatch',
    missing_fastener: 'Single fastener missing — scheduled tamping pass',
    broken_rail: 'BROKEN RAIL: Immediate halt and emergency crew dispatch',
    joint_defect: 'Minor joint gap — monitor for progressive widening',
  };

  if (btnSimulate && defectTbody) {
    btnSimulate.addEventListener('click', () => {
      const cls = classNames[Math.floor(Math.random() * classNames.length)];
      const cfg = CLASS_CONFIG[cls];
      const conf = (75 + Math.random() * 24.9).toFixed(1);
      const x1 = Math.floor(Math.random() * 600);
      const y1 = Math.floor(Math.random() * 400);
      const x2 = x1 + Math.floor(60 + Math.random() * 200);
      const y2 = y1 + Math.floor(40 + Math.random() * 150);

      idCounter++;
      totalCount++;
      if (cfg.tier === 'CRITICAL') critCount++;

      // Show modal with detection result
      openModal('New Defect Injected via detect_defects()', `{
  "class_name": "${cls}",
  "confidence": ${(conf / 100).toFixed(4)},
  "bbox": [${x1}, ${y1}, ${x2}, ${y2}],
  "priority": "${cfg.tier}",
  "reason": "${REASONS[cls]}"
}`);

      // Build table row
      const tr = document.createElement('tr');
      tr.style.animation = 'fadeIn .35s ease';
      tr.innerHTML = `
        <td><code>#DEF-${String(idCounter).padStart(3, '0')}</code></td>
        <td><span class="class-dot" style="background:${cfg.color}"></span>${cls}</td>
        <td>${conf}%</td>
        <td>${x1}, ${y1}, ${x2}, ${y2}</td>
        <td><span class="badge ${cfg.badge}">${cfg.tier}</span></td>
        <td>${REASONS[cls]}</td>
        <td><button class="tbl-btn" onclick="alert('Viewing bounding box overlay for #DEF-${String(idCounter).padStart(3, '0')} via visualizer.draw_detections()')">View</button></td>
      `;
      defectTbody.prepend(tr);

      if (critActiveNum) critActiveNum.textContent = critCount;
      if (defectCount)   defectCount.textContent   = totalCount;
    });
  }

  // Table row view buttons (initial)
  document.querySelectorAll('.tbl-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const tr = e.target.closest('tr');
      const id = tr?.querySelector('code')?.textContent || 'DEF';
      openModal(`Viewing ${id}`, `draw_detections(image_rgb, detections)
→ Corner accent brackets rendered
→ Solid color badge with confidence label
→ Auto-scaled thickness for image diagonal
→ Clamps bbox to image boundary`);
    });
  });

  if (btnExportCSV) {
    btnExportCSV.addEventListener('click', () => {
      // Build minimal CSV from table
      const rows = defectTbody?.querySelectorAll('tr') || [];
      let csv = 'defect_id,class_name,confidence,bbox,priority,reason\n';
      rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length >= 6) {
          csv += [
            cells[0].textContent.trim(),
            cells[1].textContent.trim(),
            cells[2].textContent.trim(),
            cells[3].textContent.trim(),
            cells[4].textContent.trim(),
            cells[5].textContent.trim(),
          ].map(v => `"${v}"`).join(',') + '\n';
        }
      });
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'inferail_defect_registry.csv'; a.click();
      URL.revokeObjectURL(url);
    });
  }
}

/* =================================================================
   MODAL DIALOGS
   ================================================================= */
let _modalOpen = false;

function openModal(heading, codeContent) {
  const bg         = document.getElementById('modalBg');
  const h          = document.getElementById('modalH');
  const codeBlock  = document.getElementById('modalCodeBlock');
  if (!bg) return;
  if (h) h.textContent = heading;
  if (codeBlock) codeBlock.textContent = codeContent;
  bg.classList.add('open');
  _modalOpen = true;
}

function closeModal() {
  const bg = document.getElementById('modalBg');
  if (bg) bg.classList.remove('open');
  _modalOpen = false;
}

function initModals() {
  const modalX  = document.getElementById('modalX');
  const modalBg = document.getElementById('modalBg');
  const btnLaunch  = document.getElementById('btnLaunch');
  const btnCtaDash = document.getElementById('btnCtaDash');
  const btnExploreDash = document.getElementById('btnExploreDash');
  const btnViewSample  = document.getElementById('btnViewSample');
  const btnGithub  = document.getElementById('btnGithub');
  const btnCtaGithub = document.getElementById('btnCtaGithub');

  if (modalX) modalX.addEventListener('click', closeModal);
  if (modalBg) modalBg.addEventListener('click', e => { if (e.target === modalBg) closeModal(); });
  window.addEventListener('keydown', e => { if (e.key === 'Escape' && _modalOpen) closeModal(); });

  const launchMsg = `streamlit run app.py

InfeRail Dashboard launching on http://localhost:8501
→ Upload JPG/PNG railway track image
→ CLAHE preprocessing applied (clip_limit=2.5, tile_grid=8×8)
→ YOLOv8 inference → List[DetectionResult]
→ Priority engine → PriorityReport
→ save_inspection_record() → SQLite
→ Annotated image displayed side-by-side`;

  [btnLaunch, btnCtaDash, btnExploreDash].forEach(btn => {
    if (btn) btn.addEventListener('click', () => openModal('Launch InfeRail Dashboard', launchMsg));
  });

  const sampleMsg = `preprocess_for_inference("user_track_input.jpg")

Original shape: (H, W, 3)
→ validate_image()     ✓  uint8 RGB
→ apply_clahe()        ✓  clip=2.5, LAB L-channel
→ denoise_image()      ✓  bilateral d=7, σ=50/50
→ letterbox()          ✓  224×224, pad_color=(114,114,114)
→ normalize()          ✓  ÷255 → ImageNet mean/std
→ Tensor shape:   (1, 3, 224, 224) float32`;

  if (btnViewSample) btnViewSample.addEventListener('click', () => openModal('Sample Preprocessing Run', sampleMsg));

  const githubMsg = `git clone https://github.com/team/inferail
cd inferail
pip install -r requirements.txt
streamlit run app.py`;

  [btnGithub, btnCtaGithub].forEach(btn => {
    if (btn) btn.addEventListener('click', () => openModal('Clone & Install InfeRail', githubMsg));
  });
}

/* =================================================================
   COPY COMMAND BUTTON
   ================================================================= */
function initCopyCommand() {
  const copyBtn    = document.getElementById('copyBtn');
  const installCmd = document.getElementById('installCmd');
  if (!copyBtn || !installCmd) return;

  copyBtn.addEventListener('click', () => {
    const text = 'pip install -r requirements.txt && streamlit run app.py';
    navigator.clipboard?.writeText(text).catch(() => {});
    copyBtn.innerHTML = '✓ Copied!';
    copyBtn.style.color = '#E2F952';
    setTimeout(() => {
      copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg> Copy`;
      copyBtn.style.color = '';
    }, 2000);
  });
}

/* =================================================================
   SCROLL-BASED NAV LINK HIGHLIGHTING
   ================================================================= */
function initScrollNav() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-a');

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY + 120;
    let current = '';

    sections.forEach(sec => {
      if (scrollY >= sec.offsetTop) current = sec.id;
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) link.classList.add('active');
    });
  });
}

/* =================================================================
   LIVE CONFIDENCE FLUCTUATION
   ================================================================= */
function initLiveConf() {
  const liveConf = document.getElementById('liveConf');
  if (!liveConf) return;
  setInterval(() => {
    const v = (98.5 + Math.random() * 1.4).toFixed(1);
    liveConf.textContent = v + '%';
  }, 2800);
}
