let currentFilename = null;
let currentImageData = null;

function switchStage(stageId) {
    document.querySelectorAll('.stage-section').forEach(sec => sec.classList.remove('active'));
    const target = document.getElementById(stageId);
    if (target) target.classList.add('active');

    const navMapping = {
        'stage-load': 'step-nav-1',
        'stage-preprocess': 'step-nav-2',
        'stage-features': 'step-nav-3',
        'stage-training': 'step-nav-4',
        'stage-evaluation': 'step-nav-5',
        'stage-prediction': 'step-nav-6'
    };

    document.querySelectorAll('.step-item').forEach(item => item.classList.remove('active'));
    const activeNav = document.getElementById(navMapping[stageId]);
    if (activeNav) activeNav.classList.add('active');
    window.scrollTo({ top: 120, behavior: 'smooth' });
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('status-toast');
    const toastMsg = document.getElementById('toast-message');
    if (!toast || !toastMsg) return;
    toastMsg.innerText = message;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 4000);
}

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('image', file);
    showToast('Uploading ultrasound scan...', 'info');

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.success) {
            applyLoadedImage(data);
            showToast('Scan uploaded successfully!');
        } else {
            showToast(data.error || 'Upload failed', 'error');
        }
    } catch (err) {
        showToast('Error uploading scan file', 'error');
    }
}

async function loadSampleImage(filename) {
    showToast(`Loading sample: ${filename}...`, 'info');
    try {
        const response = await fetch('/api/load-sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: filename })
        });
        const data = await response.json();
        if (data.success) {
            applyLoadedImage(data);
            showToast('Sample scan loaded!');
        }
    } catch (err) {
        showToast('Error loading sample', 'error');
    }
}

function applyLoadedImage(data) {
    currentFilename = data.filename;
    currentImageData = data;
    const previewImg = document.getElementById('loaded-image-preview');
    const placeholderText = document.getElementById('preview-placeholder-text');
    previewImg.src = data.image_b64 || data.image_url;
    previewImg.classList.remove('placeholder');
    previewImg.style.display = 'block';
    if (placeholderText) placeholderText.style.display = 'none';

    document.getElementById('image-info-box').style.display = 'block';
    document.getElementById('info-filename').innerText = data.filename;
    document.getElementById('info-dims').innerText = `${data.width} x ${data.height} px`;
    document.getElementById('img-resolution').innerText = `${data.width}x${data.height} px`;
    document.getElementById('btn-goto-preprocess').removeAttribute('disabled');
}

async function runPreprocessing() {
    if (!currentFilename) return;
    showToast('Applying preprocessors...', 'info');
    try {
        const res = await fetch('/api/preprocess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: currentFilename })
        });
        const data = await res.json();
        if (data.success) {
            const p = data.previews;
            document.getElementById('prep-img-raw').src = p.raw;
            document.getElementById('prep-img-gray').src = p.grayscale;
            document.getElementById('prep-img-denoise').src = p.denoised;
            document.getElementById('prep-img-clahe').src = p.clahe;
            document.getElementById('prep-img-segmented').src = p.segmented;
            switchStage('stage-preprocess');
            showToast('Preprocessing completed!');
        }
    } catch (err) {
        showToast('Preprocessing error', 'error');
    }
}

async function runFeatureExtraction() {
    if (!currentFilename) return;
    showToast('Computing GLCM & statistics...', 'info');
    try {
        const res = await fetch('/api/extract-features', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: currentFilename })
        });
        const data = await res.json();
        if (data.success) {
            for (const [key, val] of Object.entries(data.features_all)) {
                const el = document.getElementById(`feat-${key}`);
                if (el) el.innerText = val;
            }
            switchStage('stage-features');
            showToast('Biomarkers extracted!');
        }
    } catch (err) {
        showToast('Error extracting features', 'error');
    }
}

async function triggerModelTraining() {
    const btn = document.getElementById('btn-run-training');
    const spinner = document.getElementById('training-spinner');
    if (btn) btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');

    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            showToast(data.message);
            location.reload();
        }
    } catch (err) {
        showToast('Training failed', 'error');
    } finally {
        if (btn) btn.disabled = false;
        if (spinner) spinner.classList.add('hidden');
    }
}

async function runLivePrediction() {
    if (!currentFilename) {
        showToast('Please load an image first', 'error');
        return;
    }
    const patientId = document.getElementById('patient-id').value;
    const patientName = document.getElementById('patient-name').value;
    const patientAge = document.getElementById('patient-age').value;
    const gender = document.getElementById('patient-gender').value;
    const notes = document.getElementById('doctor-notes').value;
    const modelChoice = document.getElementById('predict-model-select').value;

    showToast('Running prediction...', 'info');

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: currentFilename,
                model: modelChoice,
                patient_id: patientId,
                patient_name: patientName,
                patient_age: patientAge,
                gender: gender,
                doctor_notes: notes
            })
        });
        const data = await res.json();
        if (data.success) {
            document.getElementById('res-display-title').innerText = data.display_title;
            document.getElementById('res-confidence').innerText = `${data.confidence}%`;
            document.getElementById('res-severity-badge').innerText = `Risk: ${data.severity}`;

            for (const cat of ['normal', 'hypothyroid', 'hyperthyroid', 'nodule']) {
                const pVal = data.probability_breakdown[cat] || 0;
                document.getElementById(`prob-val-${cat}`).innerText = `${pVal}%`;
                document.getElementById(`prob-bar-${cat}`).style.width = `${pVal}%`;
            }

            const recList = document.getElementById('res-recommendations-list');
            recList.innerHTML = '';
            data.recommendations.forEach(r => {
                const li = document.createElement('li');
                li.innerText = r;
                recList.appendChild(li);
            });

            const reportBtn = document.getElementById('btn-view-report');
            if (reportBtn) {
                reportBtn.href = `/report/${data.record_id}`;
                reportBtn.classList.remove('disabled');
            }
            showToast(`Diagnosis: ${data.display_title}`);
        }
    } catch (err) {
        showToast('Prediction failed', 'error');
    }
}

async function triggerFullDemo() {
    showToast('Running automated demo...', 'info');
    await loadSampleImage('sample_nodule_1.png');
    await new Promise(r => setTimeout(r, 600));
    await runPreprocessing();
    await new Promise(r => setTimeout(r, 800));
    await runFeatureExtraction();
    await new Promise(r => setTimeout(r, 800));
    switchStage('stage-prediction');
    await runLivePrediction();
}