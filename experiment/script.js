// script.js
// ====== Page Elements ======
const introSection = document.getElementById('intro-section');
const famSection = document.getElementById('familiarization-section');
const startSection = document.getElementById('start-section');
const loginSection = document.getElementById('login-section');
const endSection = document.getElementById('end-section');

const loginBtn = document.getElementById('loginBtn');
const loginFeedback = document.getElementById('loginFeedback');
const loginEmail = document.getElementById('loginEmail');
const loginPassword = document.getElementById('loginPassword');

// ====== Intro -> Familiarization ======
document.getElementById('introNext').addEventListener('click', () => {
  introSection.classList.add('hidden');
  famSection.classList.remove('hidden');
});

// ====== Familiarization Check ======
const famInput = document.getElementById('famInput');
const famSubmit = document.getElementById('famSubmit');
const famFeedback = document.getElementById('famFeedback');
const demoParagraph = document.getElementById('demoParagraph').innerText.trim();

famSubmit.addEventListener('click', () => {
  if (true) { // replace with: famInput.value.trim() === demoParagraph
    famFeedback.textContent = 'Correct! Moving to experiment start page...';
    famFeedback.style.color = 'green';
    setTimeout(() => {
      famSection.classList.add('hidden');
      startSection.classList.remove('hidden');
    }, 1000);
  } else {
    famFeedback.textContent = 'Text does not match exactly. Please try again.';
    famFeedback.style.color = 'red';
  }
});

// ====== Keystroke Logging (timestamps relative to Neon start) ======
let keyLogs = [];                // rows: { recording_id, timestamp (ns relative), name, type }
let recordingId = null;          // from backend
let serverStartNs = null;        // from backend (time.time_ns() at Neon start)
let clientStartPerfMid = null;   // ms (performance.now() midpoint estimate that corresponds to serverStartNs)
let currentField = 'unknown';
let _keydownHandler = null;
let _focusinHandler = null;
let _focusoutHandler = null;

function escapeCsvField(s) {
  if (s == null) return '';
  s = String(s);
  if (s.includes('"')) s = s.replace(/"/g, '""');
  if (s.includes(',') || s.includes('\n') || s.includes('"')) return `"${s}"`;
  return s;
}

function startKeyLogging() {
  // ensure mapping exists (clientStartPerfMid) — if not, fallback to using performance.now() as t0
  if (clientStartPerfMid == null) {
    // fallback: set clientStartPerfMid to current time so timestamps are relative to now
    clientStartPerfMid = performance.now();
    console.warn("clientStartPerfMid not set; falling back to local time baseline.");
  }

  keyLogs = [];

  // keydown handler (capture)
  _keydownHandler = function (e) {
    const perfNow = performance.now();
    // elapsed in ms relative to estimated Neon start time midpoint
    const elapsedMs = perfNow - clientStartPerfMid;
    // convert to ns and round
    const elapsedNs = BigInt(Math.round(elapsedMs * 1e6));
    // keep as string to avoid JSON/BigInt issues when stringifying
    const tsRelative = elapsedNs.toString();

    const fieldLabel = currentField || 'unknown';
    const name = `${fieldLabel}:${e.key} pressed`;

    keyLogs.push({
      recording_id: recordingId || 'unknown',
      timestamp: tsRelative,         // nanoseconds relative to Neon start (t=0)
      name: name,
      type: 'recording'
    });
    console.log("LOG", keyLogs[keyLogs.length - 1]);
  };

  // focus handlers to know which input is active
  _focusinHandler = function (e) {
    if (!e.target) return;
    const id = e.target.id || e.target.name || e.target.tagName.toLowerCase();
    if (id === 'loginEmail') currentField = 'username';
    else if (id === 'loginPassword') currentField = 'password';
    else currentField = id;
  };

  _focusoutHandler = function (e) {
    currentField = 'unknown';
  };

  document.addEventListener('keydown', _keydownHandler, true);
  document.addEventListener('focusin', _focusinHandler, true);
  document.addEventListener('focusout', _focusoutHandler, true);
}

function stopKeyLogging() {
  if (_keydownHandler) {
    document.removeEventListener('keydown', _keydownHandler, true);
    _keydownHandler = null;
  }
  if (_focusinHandler) {
    document.removeEventListener('focusin', _focusinHandler, true);
    _focusinHandler = null;
  }
  if (_focusoutHandler) {
    document.removeEventListener('focusout', _focusoutHandler, true);
    _focusoutHandler = null;
  }
  currentField = 'unknown';
}

// Build CSV text and trigger download locally
function downloadKeystrokesCSV() {
  if (!keyLogs || keyLogs.length === 0) {
    console.warn("No keystrokes to save.");
    return;
  }
  const header = ['recording id', 'timestamp [ns]', 'name', 'type'];
  let csv = header.join(',') + '\n';
  for (const row of keyLogs) {
    csv += [
      escapeCsvField(row.recording_id),
      escapeCsvField(row.timestamp),
      escapeCsvField(row.name),
      escapeCsvField(row.type)
    ].join(',') + '\n';
  }
  // include serverStartNs in filename if available
  const s = serverStartNs ? `_${serverStartNs}` : '';
  const filename = `keystrokes_${recordingId || Date.now()}${s}.csv`;
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  console.log(`Downloaded ${filename} (${keyLogs.length} rows)`);
}

// ====== Backend Communication for Neon control ======
async function startRecording() {
  // record client request time
  const clientRequestPerf = performance.now();
  try {
    const res = await fetch("http://127.0.0.1:5001/start_recording", { method: "POST" });
    const data = await res.json();
    const clientResponsePerf = performance.now();

    // estimate client time that corresponds to server start (midpoint)
    clientStartPerfMid = (clientRequestPerf + clientResponsePerf) / 2.0;

    if (data) {
      recordingId = data.recording_id || `local_${Date.now()}`;
      if (data.server_start_ns) {
        // server_start_ns might be large, store as string and BigInt as needed
        serverStartNs = data.server_start_ns.toString();
      } else {
        serverStartNs = null;
      }
    } else {
      recordingId = `local_${Date.now()}`;
      serverStartNs = null;
    }

    console.log("Recording started:", recordingId, "serverStartNs:", serverStartNs, "clientMidMs:", clientStartPerfMid);
    return { recordingId, serverStartNs, clientStartPerfMid };
  } catch (err) {
    console.error("Error starting recording:", err);
    // fallback: set local id and clientStartPerfMid to now
    recordingId = `local_${Date.now()}`;
    clientStartPerfMid = performance.now();
    serverStartNs = null;
    return { recordingId, serverStartNs, clientStartPerfMid };
  }
}

async function stopRecording() {
  try {
    const res = await fetch("http://127.0.0.1:5001/stop_recording", { method: "POST" });
    const data = await res.json();
    console.log("Recording stopped:", data);
  } catch (err) {
    console.error("Error stopping recording:", err);
  }
}

// Optional: send logs to backend to store server-side as well (not required)
async function sendKeystrokesToServer(save_on_server = false) {
  try {
    const payload = { logs: keyLogs, save_on_server: save_on_server };
    const res = await fetch("http://127.0.0.1:5001/save_keystrokes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    console.log("Server save response:", data);
    return data;
  } catch (err) {
    console.error("Error sending keystrokes to server:", err);
  }
}

// ====== Start Experiment Button Handler ======
document.getElementById('startExperiment').addEventListener('click', async () => {
  startSection.classList.add('hidden');

  // start backend recording and obtain id and server start time (clientStartPerfMid set inside)
  await startRecording();

  // show login and start logging (clientStartPerfMid should be set by startRecording)
  loginSection.classList.remove('hidden');
  startKeyLogging();
});

// ====== Login Handling ======
const togglePass = document.getElementById('togglePass');
togglePass.addEventListener('click', () => {
  if (loginPassword.type === 'password') {
    loginPassword.type = 'text';
    togglePass.textContent = '🙈';
  } else {
    loginPassword.type = 'password';
    togglePass.textContent = '👁️';
  }
});

async function handleLogin() {
  const username = loginEmail.value.trim();
  const password = loginPassword.value.trim();

  if (username === 'exp.user2025' && password === 'Ao4nF$Kq0!vur7?') {
    loginFeedback.textContent = 'Login successful!';
    loginFeedback.style.color = 'green';

    // stop logging & download CSV locally (timestamps are already relative to Neon start)
    stopKeyLogging();
    downloadKeystrokesCSV();

    // optional: also send to server if you want server-side saving
    // await sendKeystrokesToServer(true);

    // stop Neon recording on backend
    await stopRecording();

    setTimeout(() => {
      loginSection.classList.add('hidden');
      endSection.classList.remove('hidden');
    }, 700);
  } else {
    loginFeedback.textContent = 'Incorrect credentials.';
    loginFeedback.style.color = 'red';
    loginEmail.value = '';
    loginPassword.value = '';
  }
}

loginBtn.addEventListener('click', handleLogin);
[loginEmail, loginPassword].forEach(input => {
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') handleLogin();
  });
});
