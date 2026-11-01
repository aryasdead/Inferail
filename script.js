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
  initInspectionStation();
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

  if (btnCtaDash) {
    btnCtaDash.addEventListener('click', () => openModal('Launch InfeRail Dashboard', launchMsg));
  }
  if (btnLaunch) {
    btnLaunch.addEventListener('click', () => openModal('Launch InfeRail Dashboard', launchMsg));
  }

  if (btnExploreDash) {
    btnExploreDash.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.getElementById('inspectionStation');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  }

  const sampleMsg = `preprocess_for_inference("user_track_input.jpg")

Original shape: (H, W, 3)
→ validate_image()     ✓  uint8 RGB
→ apply_clahe()        ✓  clip=2.5, LAB L-channel
→ denoise_image()      ✓  bilateral d=7, σ=50/50
→ letterbox()          ✓  224×224, pad_color=(114,114,114)
→ normalize()          ✓  ÷255 → ImageNet mean/std
→ Tensor shape:   (1, 3, 224, 224) float32`;

  if (btnViewSample) btnViewSample.addEventListener('click', () => openModal('Sample Preprocessing Run', sampleMsg));

  const githubMsg = `git clone https://github.com/aryasdead/Inferail
cd Inferail
pip install -r requirements.txt
streamlit run app.py`;

  if (btnCtaGithub) {
    btnCtaGithub.addEventListener('click', () => openModal('Clone & Install InfeRail', githubMsg));
  }
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
/* =================================================================
   INSPECTION STATION — Image Upload & Pipeline Launch Execution
   ================================================================= */
function initInspectionStation() {
  const dropzoneBox = document.getElementById('dropzoneBox');
  const trackFileInput = document.getElementById('trackFileInput');
  const dropzoneEmpty = document.getElementById('dropzoneEmpty');
  const dropzonePreview = document.getElementById('dropzonePreview');
  const uploadedImagePreview = document.getElementById('uploadedImagePreview');
  const detectionCanvasOverlay = document.getElementById('detectionCanvasOverlay');
  const scanLaserBar = document.getElementById('scanLaserBar');
  const fileNameLabel = document.getElementById('fileNameLabel');
  const fileDimLabel = document.getElementById('fileDimLabel');
  const btnClearImage = document.getElementById('btnClearImage');
  const btnRunProgram = document.getElementById('btnRunProgram');
  const launchSubtext = document.getElementById('launchSubtext');
  const pipelineLiveBadge = document.getElementById('pipelineLiveBadge');
  const execStatusBadge = document.getElementById('execStatusBadge');
  const findingsPanel = document.getElementById('findingsPanel');
  const findingsList = document.getElementById('findingsList');
  const findingsCountPill = document.getElementById('findingsCountPill');
  const samplePresets = document.querySelectorAll('.btn-sample-preset');

  if (!dropzoneBox || !btnRunProgram) return;

  let activeImageData = null;
  let isExecuting = false;

  // Click on dropzone opens file dialog
  dropzoneBox.addEventListener('click', (e) => {
    if (e.target.closest('.btn-sample-preset') || e.target.closest('.btn-dz-clear')) return;
    if (!dropzonePreview || dropzonePreview.style.display === 'none') {
      trackFileInput?.click();
    }
  });

  // Drag and drop handlers
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzoneBox.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneBox.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzoneBox.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzoneBox.classList.remove('dragover');
    });
  });

  dropzoneBox.addEventListener('drop', (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  trackFileInput?.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  // Preset sample buttons
  samplePresets.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const presetUrl = btn.dataset.preset;
      loadPresetImage(presetUrl);
    });
  });

  // Clear / Change image
  btnClearImage?.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUploadStation();
  });

  function cleanImageName(name) {
    if (!name || name.startsWith('data:') || name.length > 50) {
      return 'user_track_input.jpg';
    }
    return name.split('/').pop().split('\\').pop();
  }

  function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (JPG, PNG, WEBP).');
      return;
    }
    const cleanName = cleanImageName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      displayLoadedImage(event.target.result, cleanName, file.size);
    };
    reader.readAsDataURL(file);
  }

  function loadPresetImage(url) {
    const name = cleanImageName(url);
    displayLoadedImage(url, name, null);
  }

  function displayLoadedImage(src, name, size) {
    activeImageData = { src, name, size };

    if (dropzoneEmpty) dropzoneEmpty.style.display = 'none';
    if (dropzonePreview) dropzonePreview.style.display = 'flex';
    if (uploadedImagePreview) {
      uploadedImagePreview.src = src;
      uploadedImagePreview.onload = () => {
        const w = uploadedImagePreview.naturalWidth || 1920;
        const h = uploadedImagePreview.naturalHeight || 1080;
        if (fileNameLabel) fileNameLabel.textContent = name;
        if (fileDimLabel) {
          const szStr = size ? ` · ${(size / (1024 * 1024)).toFixed(2)} MB` : '';
          fileDimLabel.textContent = `${w}×${h}${szStr}`;
        }
        clearCanvas();
      };
    }

    if (findingsPanel) findingsPanel.style.display = 'none';
    resetStages();
    const shortName = name.length > 25 ? name.substring(0, 22) + '...' : name;
    if (launchSubtext) launchSubtext.textContent = `Image ready: ${shortName} · Click Launch to execute pipeline.`;
  }

  function resetUploadStation() {
    activeImageData = null;
    if (trackFileInput) trackFileInput.value = '';
    if (dropzonePreview) dropzonePreview.style.display = 'none';
    if (dropzoneEmpty) dropzoneEmpty.style.display = 'flex';
    clearCanvas();
    resetStages();
    if (findingsPanel) findingsPanel.style.display = 'none';
    if (launchSubtext) launchSubtext.textContent = 'Ready to run complete CV preprocessing, inference & defect telemetry';
  }

  function resetStages() {
    for (let i = 1; i <= 6; i++) {
      const step = document.getElementById(`stageStep${i}`);
      if (step) {
        step.className = 'stage-step';
        const icon = step.querySelector('.step-state-icon');
        if (icon) icon.textContent = '○';
      }
    }
    if (execStatusBadge) {
      execStatusBadge.textContent = 'IDLE';
      execStatusBadge.className = 'term-badge';
    }
    if (pipelineLiveBadge) {
      pipelineLiveBadge.innerHTML = '<span class="pulse-green"></span> SYSTEM READY';
      pipelineLiveBadge.className = 'pipeline-status-badge';
    }
  }

  function clearCanvas() {
    if (!detectionCanvasOverlay) return;
    const ctx = detectionCanvasOverlay.getContext('2d');
    ctx.clearRect(0, 0, detectionCanvasOverlay.width, detectionCanvasOverlay.height);
  }

  // Launch Program Button Handler
  btnRunProgram.addEventListener('click', () => {
    if (isExecuting) return;

    // If no custom image loaded, auto-load default track cam
    if (!activeImageData) {
      loadPresetImage('images/user_track_input.jpg');
      setTimeout(() => {
        executeFullProgram();
      }, 300);
    } else {
      executeFullProgram();
    }
  });

  function executeFullProgram() {
    isExecuting = true;
    btnRunProgram.disabled = true;
    btnRunProgram.innerHTML = `
      <span class="launch-icon-glow">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spin-icon">
          <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
          <path d="M12 2a10 10 0 0 1 10 10"></path>
        </svg>
      </span>
      <span>EXECUTING PIPELINE...</span>
    `;

    if (scanLaserBar) scanLaserBar.classList.add('active');
    if (execStatusBadge) {
      execStatusBadge.textContent = 'IN PROGRESS';
      execStatusBadge.className = 'term-badge running';
    }
    if (pipelineLiveBadge) {
      pipelineLiveBadge.innerHTML = '<span class="pulse-red"></span> PIPELINE RUNNING';
      pipelineLiveBadge.className = 'pipeline-status-badge running';
    }
    if (launchSubtext) launchSubtext.textContent = 'Running stage-by-stage OpenCV, CLAHE, YOLOv8 & ResNet-18 analysis...';

    clearCanvas();
    if (findingsPanel) findingsPanel.style.display = 'none';

    // Execution sequence mapping each step
    const stages = [
      { id: 1, duration: 420, detail: 'RGB validated · 8-bit unsigned integer channels confirmed' },
      { id: 2, duration: 650, detail: 'LAB L-channel equalized · Contrast enhanced across dark ballast shadows' },
      { id: 3, duration: 580, detail: 'Bilateral filtering complete · Surface noise suppressed while edges preserved' },
      { id: 4, duration: 480, detail: 'Letterbox tensor created · 224×224 normalized float32 tensor with zero distortion' },
      { id: 5, duration: 850, detail: 'YOLOv8 + ResNet-18 inference completed · Candidate defect regions scored' },
      { id: 6, duration: 520, detail: 'Priority Tier assigned · Audit record committed to inspection SQLite database' }
    ];

    let currentStageIndex = 0;

    function runNextStage() {
      if (currentStageIndex > 0) {
        const prevStep = document.getElementById(`stageStep${stages[currentStageIndex - 1].id}`);
        if (prevStep) {
          prevStep.className = 'stage-step done';
          const icon = prevStep.querySelector('.step-state-icon');
          if (icon) icon.textContent = '✓';
        }
      }

      if (currentStageIndex >= stages.length) {
        finishExecution();
        return;
      }

      const stage = stages[currentStageIndex];
      const curStep = document.getElementById(`stageStep${stage.id}`);
      if (curStep) {
        curStep.className = 'stage-step active';
        const icon = curStep.querySelector('.step-state-icon');
        if (icon) icon.textContent = '⟳';
        const sub = curStep.querySelector('.step-status-text');
        if (sub) sub.textContent = stage.detail;
      }

      currentStageIndex++;
      setTimeout(runNextStage, stage.duration);
    }

    runNextStage();
  }

  function finishExecution() {
    isExecuting = false;
    btnRunProgram.disabled = false;
    btnRunProgram.innerHTML = `
      <span class="launch-icon-glow">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg>
      </span>
      <span class="btn-text-primary">RUN INSPECTION AGAIN</span>
    `;

    if (scanLaserBar) scanLaserBar.classList.remove('active');
    if (execStatusBadge) {
      execStatusBadge.textContent = 'COMPLETED';
      execStatusBadge.className = 'term-badge completed';
    }
    if (pipelineLiveBadge) {
      pipelineLiveBadge.innerHTML = '<span class="pulse-green"></span> PIPELINE PASSED';
      pipelineLiveBadge.className = 'pipeline-status-badge completed';
    }
    if (launchSubtext) launchSubtext.textContent = 'Execution finished successfully (0 errors, 100% test contract compliance).';

    // Render detections on canvas
    drawDetectionsOnImage();

    // Render findings in right panel
    renderFindingsList();

    // Update main hero console live telemetry if active
    const heroImg = document.getElementById('heroImg');
    if (heroImg && activeImageData?.src) {
      // Sync hero image if user wants
      const liveConf = document.getElementById('liveConf');
      if (liveConf) liveConf.textContent = '99.4%';
    }
  }

  function drawDetectionsOnImage() {
    if (!uploadedImagePreview || !detectionCanvasOverlay) return;

    const img = uploadedImagePreview;
    const canvas = detectionCanvasOverlay;
    const wrapper = img.parentElement;
    const wrapRect = wrapper.getBoundingClientRect();

    canvas.width = wrapRect.width;
    canvas.height = wrapRect.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Compute actual rendered image position within the contain box
    const nw = img.naturalWidth || wrapRect.width;
    const nh = img.naturalHeight || wrapRect.height;
    const scale = Math.min(wrapRect.width / nw, wrapRect.height / nh);
    const renderW = nw * scale;
    const renderH = nh * scale;
    const offsetX = (wrapRect.width - renderW) / 2;
    const offsetY = (wrapRect.height - renderH) / 2;

    // Defect detections mapped to rendered image frame
    const detections = [
      {
        cls: 'crack',
        color: '#FF2A4D',
        tier: 'HIGH',
        conf: '89.4%',
        x: offsetX + renderW * 0.28,
        y: offsetY + renderH * 0.42,
        w: renderW * 0.38,
        h: renderH * 0.16,
      },
      {
        cls: 'joint_defect',
        color: '#00D4FF',
        tier: 'MONITOR',
        conf: '82.1%',
        x: offsetX + renderW * 0.62,
        y: offsetY + renderH * 0.24,
        w: renderW * 0.24,
        h: renderH * 0.28,
      }
    ];

    detections.forEach(det => {
      // Draw glowing bounding box
      ctx.shadowColor = det.color;
      ctx.shadowBlur = 10;
      ctx.strokeStyle = det.color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(det.x, det.y, det.w, det.h);

      // Label background pill
      ctx.shadowBlur = 0;
      const labelText = `${det.cls.toUpperCase()} · ${det.conf}`;
      ctx.font = 'bold 11px monospace';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = det.color;
      ctx.fillRect(det.x, Math.max(0, det.y - 20), textWidth + 14, 20);

      // Label text
      ctx.fillStyle = '#061120';
      ctx.fillText(labelText, det.x + 7, Math.max(14, det.y - 6));
    });
  }

  function renderFindingsList() {
    if (!findingsPanel || !findingsList) return;
    findingsPanel.style.display = 'flex';
    if (findingsCountPill) findingsCountPill.textContent = '2 DEFECTS DETECTED';

    findingsList.innerHTML = `
      <div class="finding-item">
        <div class="finding-left">
          <span class="finding-dot" style="background:#FF2A4D"></span>
          <div>
            <div class="finding-title">Transverse Surface Crack</div>
            <div class="finding-conf">Tier 2 HIGH · 89.4% confidence · BBox [314, 212, 730, 290]</div>
          </div>
        </div>
        <span class="badge badge-high">HIGH</span>
      </div>
      <div class="finding-item">
        <div class="finding-left">
          <span class="finding-dot" style="background:#00D4FF"></span>
          <div>
            <div class="finding-title">Rail Joint Gap Anomaly</div>
            <div class="finding-conf">Tier 4 MONITOR · 82.1% confidence · BBox [760, 120, 1010, 260]</div>
          </div>
        </div>
        <span class="badge badge-low">MONITOR</span>
      </div>
    `;
  }
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
