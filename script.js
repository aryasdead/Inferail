/**
 * InfeRail AI — Definitive Script
 * Powers: background particle canvas, hero image switcher,
 * modal dialogs, nav scroll, and the live inspection station pipeline.
 */

document.addEventListener('DOMContentLoaded', () => {
  initBgCanvas();
  initHeroImageSwitcher();
  initInspectionStation();
  initModals();
  initScrollNav();
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
    'annotated_user_track.jpg':    { t1: 'CRACK DETECTED', d1: 'crack · 89.0% conf · Severity: High (Sample frame)', t2: 'JOINT DEFECT DETECTED', d2: 'joint_defect · 78.0% · Streamlit YOLOv8 <!-- TODO: verify --> (Sample frame)' },
    'side_by_side_user_track.jpg': { t1: 'DUAL INSPECTION', d1: 'Original vs CLAHE/Annotated (Sample frame)', t2: 'DEFECT COUNT', d2: '3 Flaws Flagged (Sample frame)' },
    'enhanced.jpg':                { t1: 'CLAHE ENHANCED', d1: 'Local contrast normalized (clip=2.0)', t2: 'BALLAST NOISE', d2: 'Bilateral filtered' },
    'track_sample_crack_shadow.jpg': { t1: 'CRACK + SHADOW', d1: 'crack under low ambient · 92.4% (Sample frame)', t2: '', d2: '' },
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
  const btnExploreDash = document.getElementById('btnExploreDash');

  if (modalX) modalX.addEventListener('click', closeModal);
  if (modalBg) modalBg.addEventListener('click', e => { if (e.target === modalBg) closeModal(); });
  window.addEventListener('keydown', e => { if (e.key === 'Escape' && _modalOpen) closeModal(); });

  if (btnExploreDash) {
    btnExploreDash.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.getElementById('inspectionStation');
      if (!target) return;
      if (window.__lenis) {
        window.__lenis.scrollTo(target);
      } else {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  }
}

/* =================================================================
   SCROLL-BASED NAV LINK HIGHLIGHTING
   ================================================================= */
function initScrollNav() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-a');

  const onScroll = () => {
    const scrollY = window.scrollY + 120;
    let current = '';
    sections.forEach(sec => {
      if (scrollY >= sec.offsetTop) current = sec.id;
    });
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) link.classList.add('active');
    });
  };

  // Use Lenis scroll event when available, fall back to window scroll
  if (window.__lenis) {
    window.__lenis.on('scroll', onScroll);
  } else {
    window.addEventListener('scroll', onScroll);
    // Also try to attach once Lenis is ready (Lenis init happens after DOMContentLoaded)
    window.addEventListener('lenis-ready', () => {
      window.removeEventListener('scroll', onScroll);
      window.__lenis.on('scroll', onScroll);
    });
  }
}

/* =================================================================
   SEVERITY COLOR MAP — canonical colors by severity string
   ================================================================= */
const SEVERITY_COLORS = {
  CRITICAL: '#D600FF',
  HIGH:     '#FF2A4D',
  MEDIUM:   '#FFAA00',
  NORMAL:   '#00D4FF',
  MONITOR:  '#00D4FF',
  OFFLINE:  '#FFAA00',
};

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
  const viewSegCtrl = document.getElementById('viewSegCtrl');
  const pipelineProgressFill = document.getElementById('pipelineProgressFill');

  if (!dropzoneBox || !btnRunProgram) return;

  let activeImageData = null;
  let isExecuting = false;

  // MutationObserver: watch stage-step class changes → update progress bar
  if (pipelineProgressFill) {
    const observer = new MutationObserver(() => {
      let done = 0;
      for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`stageStep${i}`);
        if (step && step.classList.contains('done')) done++;
      }
      pipelineProgressFill.style.width = `${Math.round((done / 6) * 100)}%`;
    });
    for (let i = 1; i <= 6; i++) {
      const step = document.getElementById(`stageStep${i}`);
      if (step) observer.observe(step, { attributes: true, attributeFilter: ['class'] });
    }
  }

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

  // View segmented control (Original / CLAHE / Annotated)
  if (viewSegCtrl) {
    viewSegCtrl.querySelectorAll('.view-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        viewSegCtrl.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.dataset.view;
        const last = window.__inferailLast;
        if (!uploadedImagePreview) return;
        if (view === 'clahe' && last && last.enhanced_image_b64) {
          uploadedImagePreview.src = last.enhanced_image_b64;
        } else if (view === 'annotated' && last && last.annotated_image_b64) {
          uploadedImagePreview.src = last.annotated_image_b64;
        } else {
          // Original
          if (activeImageData) uploadedImagePreview.src = activeImageData.src;
        }
      });
    });
  }

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
      displayLoadedImage(event.target.result, cleanName, file.size, file, null);
    };
    reader.readAsDataURL(file);
  }

  function loadPresetImage(url) {
    const name = cleanImageName(url);
    displayLoadedImage(url, name, null, null, url);
  }

  function displayLoadedImage(src, name, size, rawFile = null, presetUrl = null) {
    activeImageData = { src, name, size, rawFile, presetUrl };

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
    if (viewSegCtrl) viewSegCtrl.style.display = 'none';
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
    if (viewSegCtrl) viewSegCtrl.style.display = 'none';
    if (launchSubtext) launchSubtext.textContent = 'Ready to run complete CV preprocessing, inference & defect telemetry';
  }

  function resetStages() {
    for (let i = 1; i <= 6; i++) {
      const step = document.getElementById(`stageStep${i}`);
      if (step) {
        step.className = 'stage-step';
        const icon = step.querySelector('.step-state-icon');
        if (icon) icon.textContent = '○';
        const timing = step.querySelector('.step-timing');
        if (timing) timing.textContent = '';
      }
    }
    if (pipelineProgressFill) pipelineProgressFill.style.width = '0%';
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

  let lastInspectionResult = null;

  async function executeFullProgram() {
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
    if (launchSubtext) launchSubtext.textContent = 'Executing OpenCV, CLAHE, Bilateral Denoising & ResNet-18 model...';

    clearCanvas();
    if (findingsPanel) findingsPanel.style.display = 'none';
    if (viewSegCtrl) viewSegCtrl.style.display = 'none';
    // Remove any previous offline banner
    document.getElementById('offlineWarningBanner')?.remove();

    // Stage label definitions (truthful — matches actual server.py pipeline)
    const stages = [
      { id: 1, detail: 'RGB validated · 8-bit unsigned integer channels confirmed' },
      { id: 2, detail: 'LAB L-channel equalized · Contrast enhanced across dark ballast shadows' },
      { id: 3, detail: 'Bilateral filtering complete · Surface noise suppressed while edges preserved' },
      { id: 4, detail: 'Letterbox tensor created · 224×224 normalized float32 tensor with zero distortion' },
      { id: 5, detail: 'ResNet-18 CrackClassifier inference · Defect probability and feature activations evaluated' },
      { id: 6, detail: 'Priority tier assigned · Inspection record flagged for maintenance queue' }
    ];

    function markStage(stageId, state, customDetail = null, timingMs = null) {
      const step = document.getElementById(`stageStep${stageId}`);
      if (!step) return;
      step.className = `stage-step ${state}`;
      const icon = step.querySelector('.step-state-icon');
      if (icon) {
        if (state === 'done') icon.textContent = '✓';
        else if (state === 'active') icon.textContent = '⟳';
        else icon.textContent = '○';
      }
      if (customDetail) {
        const sub = step.querySelector('.step-status-text');
        if (sub) sub.textContent = customDetail;
      }
      if (timingMs != null) {
        const timing = step.querySelector('.step-timing');
        if (timing) timing.textContent = `${timingMs}ms`;
      }
    }

    // Prepare FormData payload for backend
    const formData = new FormData();
    if (activeImageData && activeImageData.rawFile) {
      formData.append('file', activeImageData.rawFile);
    } else if (activeImageData && activeImageData.presetUrl) {
      formData.append('preset_path', activeImageData.presetUrl);
    } else {
      formData.append('preset_path', 'images/user_track_input.jpg');
    }

    // Animate initial stages
    markStage(1, 'active', stages[0].detail);
    setTimeout(() => { markStage(1, 'done'); markStage(2, 'active', stages[1].detail); }, 220);
    setTimeout(() => { markStage(2, 'done'); markStage(3, 'active', stages[2].detail); }, 450);
    setTimeout(() => { markStage(3, 'done'); markStage(4, 'active', stages[3].detail); }, 680);
    setTimeout(() => { markStage(4, 'done'); markStage(5, 'active', stages[4].detail); }, 900);

    try {
      const response = await fetch('/api/inspect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      lastInspectionResult = data;
      window.__inferailLast = data;

      const t = data.timings_ms || {};

      // Back-fill real timings for early stages now that API has returned
      markStage(1, 'done', stages[0].detail, t.validation_ms ?? null);
      markStage(2, 'done', stages[1].detail, t.clahe_ms ?? null);
      markStage(3, 'done', stages[2].detail, t.denoising_ms ?? null);
      markStage(4, 'done', stages[3].detail, t.letterboxing_ms ?? null);

      // Complete stages with real timing details
      markStage(5, 'done',
        `ResNet-18 inference: ${t.inference_ms ?? '?'}ms · Score: ${(data.crack_probability * 100).toFixed(1)}%`,
        t.inference_ms ?? null
      );
      markStage(6, 'active', stages[5].detail);
      setTimeout(() => {
        markStage(6, 'done',
          `${data.priority_tier} · Inspection record flagged`,
          t.postprocessing_ms ?? null
        );
        finishExecution(data);
      }, 350);

    } catch (err) {
      console.warn('Backend API unavailable or error occurred, using offline simulation fallback:', err);
      // Offline simulation fallback — NOT a real result
      setTimeout(() => {
        markStage(5, 'done', 'OFFLINE — no real inference performed');
        markStage(6, 'done', 'OFFLINE — no real priority assigned');
        const fallbackData = {
          prediction: 'cracked',
          confidence: 0.0,
          crack_probability: 0.0,
          severity: 'OFFLINE',
          priority_tier: 'OFFLINE SIMULATION',
          action_plan: '⚠ OFFLINE SIMULATION — not a real result',
          defect_count: 0,
          timings_ms: { total_ms: 0 },
          detections: [],
          findings: [{
            id: 'OFFLINE-01',
            title: 'OFFLINE SIMULATION — not a real result',
            description: 'The server is unreachable. Start `python server.py` and re-run.',
            severity: 'OFFLINE',
          }],
          isOffline: true,
        };
        finishExecution(fallbackData);
      }, 1100);
    }
  }

  function finishExecution(data) {
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

    const isCracked = data.prediction === 'cracked' && !data.isOffline;

    if (execStatusBadge) {
      if (data.isOffline) {
        execStatusBadge.textContent = 'OFFLINE';
        execStatusBadge.className = 'term-badge running';
      } else {
        execStatusBadge.textContent = isCracked ? 'DEFECT DETECTED' : 'TRACK NORMAL';
        execStatusBadge.className = isCracked ? 'term-badge completed error' : 'term-badge completed';
      }
    }
    if (pipelineLiveBadge) {
      if (data.isOffline) {
        pipelineLiveBadge.innerHTML = '<span class="pulse-red"></span> OFFLINE';
        pipelineLiveBadge.className = 'pipeline-status-badge running';
      } else {
        pipelineLiveBadge.innerHTML = isCracked
          ? '<span class="pulse-red"></span> DEFECT DETECTED'
          : '<span class="pulse-green"></span> PIPELINE PASSED';
        pipelineLiveBadge.className = isCracked
          ? 'pipeline-status-badge running'
          : 'pipeline-status-badge completed';
      }
    }
    if (launchSubtext) {
      if (data.isOffline) {
        launchSubtext.textContent = '⚠ OFFLINE SIMULATION — not a real result. Start server.py and re-run.';
      } else {
        const confStr = (data.confidence * 100).toFixed(1);
        const totalTime = data.timings_ms?.total_ms || 150;
        launchSubtext.textContent = `Completed in ${totalTime}ms · Prediction: ${data.prediction.toUpperCase()} (${confStr}% conf) · ${data.action_plan}`;
      }
    }

    // Inject offline warning banner above findings panel
    if (data.isOffline) {
      const existing = document.getElementById('offlineWarningBanner');
      if (!existing && findingsPanel) {
        const banner = document.createElement('div');
        banner.id = 'offlineWarningBanner';
        banner.className = 'offline-warning-banner';
        banner.innerHTML = '⚠ OFFLINE SIMULATION — not a real result. The backend server is unreachable.';
        findingsPanel.parentElement.insertBefore(banner, findingsPanel);
      }
    }

    // Render detections on canvas (only when online)
    if (!data.isOffline) {
      drawDetectionsOnImage(data.detections || [], data.severity);
    }

    // Render findings in panel
    renderFindingsList(data);

    // Show view segmented control (only when real data with images available)
    if (!data.isOffline && viewSegCtrl && (data.enhanced_image_b64 || data.annotated_image_b64)) {
      viewSegCtrl.style.display = 'flex';
      // Reset to original view
      viewSegCtrl.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      const origBtn = viewSegCtrl.querySelector('[data-view="original"]');
      if (origBtn) origBtn.classList.add('active');
    }

    // Update live telemetry confidence from real data
    const liveConf = document.getElementById('liveConf');
    if (liveConf && !data.isOffline) {
      liveConf.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    }
  }

  function drawDetectionsOnImage(detections, overallSeverity) {
    if (!uploadedImagePreview || !detectionCanvasOverlay) return;

    const img = uploadedImagePreview;
    const canvas = detectionCanvasOverlay;
    const wrapper = img.parentElement;
    const wrapRect = wrapper.getBoundingClientRect();

    canvas.width = wrapRect.width;
    canvas.height = wrapRect.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!detections || detections.length === 0) return;

    // Compute actual rendered image position within the contain box
    const nw = img.naturalWidth || wrapRect.width;
    const nh = img.naturalHeight || wrapRect.height;
    const scale = Math.min(wrapRect.width / nw, wrapRect.height / nh);
    const renderW = nw * scale;
    const renderH = nh * scale;
    const offsetX = (wrapRect.width - renderW) / 2;
    const offsetY = (wrapRect.height - renderH) / 2;

    detections.forEach(det => {
      let x, y, w, h;
      if (det.norm_bbox) {
        x = offsetX + det.norm_bbox[0] * renderW;
        y = offsetY + det.norm_bbox[1] * renderH;
        w = (det.norm_bbox[2] - det.norm_bbox[0]) * renderW;
        h = (det.norm_bbox[3] - det.norm_bbox[1]) * renderH;
      } else if (det.bbox) {
        x = offsetX + (det.bbox[0] / nw) * renderW;
        y = offsetY + (det.bbox[1] / nh) * renderH;
        w = ((det.bbox[2] - det.bbox[0]) / nw) * renderW;
        h = ((det.bbox[3] - det.bbox[1]) / nh) * renderH;
      } else {
        return;
      }

      // Color by severity string, not by backend-provided color field
      const sev = det.tier || overallSeverity || 'NORMAL';
      const color = SEVERITY_COLORS[sev] || '#00D4FF';

      // Thin square-cornered stroke (no shadow)
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, y, w, h);

      // Flat label (no shadow)
      const labelText = det.label || `${(det.cls || 'defect').toUpperCase()} · ${det.tier || ''}`;
      ctx.font = 'bold 11px monospace';
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 20), textWidth + 14, 20);

      ctx.shadowBlur = 0;
      ctx.fillStyle = '#061120';
      ctx.fillText(labelText, x + 7, Math.max(14, y - 6));
    });
  }

  function renderFindingsList(data) {
    if (!findingsPanel || !findingsList) return;
    findingsPanel.style.display = 'flex';

    const count = data.defect_count ?? (data.detections ? data.detections.length : 0);
    if (findingsCountPill) {
      if (data.isOffline) {
        findingsCountPill.textContent = 'OFFLINE — no real result';
      } else {
        findingsCountPill.textContent = count > 0 ? `${count} DEFECT${count > 1 ? 'S' : ''} DETECTED` : '0 DEFECTS (TRACK INTACT)';
      }
    }

    if (data.findings && data.findings.length > 0) {
      findingsList.innerHTML = data.findings.map(f => {
        // Severity color by string
        const dotColor = SEVERITY_COLORS[f.severity] || '#00D4FF';
        let badgeClass = 'badge-low';
        if (f.severity === 'CRITICAL') badgeClass = 'badge-crit';
        else if (f.severity === 'HIGH') badgeClass = 'badge-high';
        else if (f.severity === 'MEDIUM') badgeClass = 'badge-med';
        else if (f.severity === 'OFFLINE') badgeClass = 'badge-med';
        return `
          <div class="finding-item">
            <div class="finding-left">
              <span class="finding-dot" style="background:${dotColor}"></span>
              <div>
                <div class="finding-title">${f.title}</div>
                <div class="finding-conf">${f.description}</div>
              </div>
            </div>
            <span class="badge ${badgeClass}">${f.severity}</span>
          </div>
        `;
      }).join('');
    } else {
      findingsList.innerHTML = `
        <div class="finding-item">
          <div class="finding-left">
            <span class="finding-dot" style="background:#00D4FF"></span>
            <div>
              <div class="finding-title">Track Condition Nominal</div>
              <div class="finding-conf">Tier 4 MONITOR · Clean rail surface with no structural fatigue</div>
            </div>
          </div>
          <span class="badge badge-low">NOMINAL</span>
        </div>
      `;
    }
  }
}

/* =================================================================
   MOTION & SMOOTH SCROLL — Lenis + GSAP ScrollTrigger
   ================================================================= */
(function initMotion() {
  // Guard: only run if both Lenis and GSAP are loaded from CDN
  if (typeof Lenis === 'undefined' || typeof gsap === 'undefined') return;

  // Initialise Lenis smooth scroll
  const lenis = new Lenis({ lerp: 0.08 });
  window.__lenis = lenis;

  // Dispatch custom event so initScrollNav can upgrade its listener
  window.dispatchEvent(new Event('lenis-ready'));

  // Connect Lenis to GSAP ticker
  if (typeof ScrollTrigger !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    // Scroll-triggered fade-in for cards and pipeline steps
    gsap.utils.toArray('.glass-card, .stage-step, .flow-step').forEach((el) => {
      gsap.from(el, {
        opacity: 0,
        y: 28,
        duration: 0.55,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: el,
          start: 'top 92%',
          toggleActions: 'play none none none',
        },
      });
    });
  } else {
    // GSAP loaded but ScrollTrigger not available — just run Lenis with rAF
    function rafLoop(time) {
      lenis.raf(time);
      requestAnimationFrame(rafLoop);
    }
    requestAnimationFrame(rafLoop);
  }
})();
