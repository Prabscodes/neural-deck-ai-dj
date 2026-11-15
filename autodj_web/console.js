"use strict";

const find = id => document.getElementById(id);
const form = find("dj-form");
const input = find("request");
const startButton = find("live-start");
const steerButton = find("live-steer");
const pauseButton = find("live-pause");
const stopButton = find("live-stop");
const volume = find("volume");
const muteButton = find("mute-live");
const scope = find("visual");
const monitor = find("mix-monitor");
const bands = [["bass", 40, 250], ["mids", 250, 2000], ["highs", 2000, 12000]];
const canvas = find("spectrum");
const context2d = canvas.getContext("2d");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const presets = [...document.querySelectorAll(".chip[data-prompt]")];
let live = null;
let starting = false;
let muted = false;
let scopeEnergy = -1;
let paintedIdle = false;

const setText = (element, value) => {
  if (element.textContent !== value) element.textContent = value;
};

function updatePresets() {
  for (const button of presets) button.setAttribute("aria-pressed", String(input.value === button.dataset.prompt));
}
input.addEventListener("input", updatePresets);
for (const button of presets) {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    updatePresets();
    input.focus();
  });
}
updatePresets();

function activity(message, state = "") {
  find("status").dataset.state = state;
  setText(find("status"), message);
}

function indicator(state, label) {
  const badge = find("console-state");
  badge.dataset.state = state;
  setText(badge.lastElementChild, label);
  if (scope.dataset.state !== state) {
    scope.dataset.state = state;
    monitor.dataset.state = state;
  }
}

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {cache: "no-store"} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw Error(typeof result.detail === "string" ? result.detail : "Please try again.");
  return result;
}

function syncControls() {
  startButton.hidden = !!live;
  startButton.disabled = starting;
  steerButton.hidden = !live;
  find("live-controls").hidden = !live;
}

function setVolume() {
  const silent = muted || Number(volume.value) === 0;
  muteButton.setAttribute("aria-pressed", String(silent));
  muteButton.setAttribute("aria-label", silent ? "Unmute audio" : "Mute audio");
  if (live) live.gain.gain.setTargetAtTime(silent ? 0 : Number(volume.value) / 100, live.context.currentTime, .015);
}
muteButton.addEventListener("click", () => {
  if (Number(volume.value) === 0) {
    volume.value = 100;
    muted = false;
  } else {
    muted = !muted;
  }
  setVolume();
});
volume.addEventListener("input", () => {
  muted = false;
  setVolume();
});

function renderUpcoming(session) {
  const next = session.segments.filter(segment => segment.seq > session.lastPlayingSeq).slice(0, 2);
  const signature = JSON.stringify(next.map(segment => [segment.seq, segment.title]));
  if (signature === session.upcoming) return;
  session.upcoming = signature;
  find("next-section").hidden = next.length === 0;
  find("tracks").replaceChildren(...next.map(segment => {
    const row = document.createElement("li");
    row.textContent = segment.title;
    return row;
  }));
}

function fitSpectrum() {
  const ratio = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  canvas.height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  drawSpectrum(live, true);
}

function drawIdleTrace(width, height) {
  for (const [offset, opacity] of [[0, .36], [.9, .21], [1.8, .14]]) {
    context2d.globalAlpha = opacity;
    context2d.beginPath();
    for (let i = 0; i <= 90; i++) {
      const x = i / 90;
      const envelope = Math.sin(Math.PI * x) ** 2;
      const wave = Math.sin(x * 15 + offset) * .6 + Math.sin(x * 25 - offset) * .4;
      const y = height * (.5 + .21 * envelope * wave);
      if (i === 0) context2d.moveTo(0, y);
      else context2d.lineTo(x * width, y);
    }
    context2d.stroke();
  }
}

function updateMonitor(session) {
  for (const [name, low, high] of bands) {
    let value = 0;
    if (session) {
      const binHz = session.context.sampleRate / session.analyser.fftSize;
      const start = Math.max(1, Math.ceil(low / binHz));
      const end = Math.min(session.bins.length, Math.ceil(high / binHz));
      let sum = 0;
      for (let i = start; i < end; i++) sum += session.bins[i] ** 2;
      value = Math.sqrt(sum / Math.max(1, end - start)) / 255;
    }
    const rounded = (Math.round(value * 25) / 25).toString();
    const property = `--level-${name}`;
    if (monitor.style.getPropertyValue(property) !== rounded) monitor.style.setProperty(property, rounded);
  }
}

function drawSpectrum(session, force = false) {
  if (!context2d) return;
  const active = !!(session?.current && !reducedMotion.matches);
  if (!active && paintedIdle && !force) return;
  paintedIdle = !active;
  const {width, height} = canvas;
  context2d.clearRect(0, 0, width, height);

  let energy = 0;
  if (active) {
    session.analyser.getByteFrequencyData(session.bins);
    const bass = session.bins[1] + session.bins[2] + session.bins[3] + session.bins[4];
    energy = Math.round(bass * 20 / (4 * 255)) / 20;
  }
  updateMonitor(active ? session : null);
  if (scopeEnergy !== energy) {
    scopeEnergy = energy;
    scope.style.setProperty("--scope-energy", energy);
  }

  const ramp = context2d.createLinearGradient(0, 0, width, 0);
  ramp.addColorStop(0, "#a3fbdf");
  ramp.addColorStop(.55, "#a7def9");
  ramp.addColorStop(1, "#c2bdff");
  context2d.fillStyle = ramp;
  context2d.strokeStyle = ramp;
  if (!active) {
    context2d.lineWidth = 1;
    drawIdleTrace(width, height);
    context2d.globalAlpha = 1;
    return;
  }

  const count = width > 420 ? 62 : 46;
  const stride = width / count;
  for (let index = 0; index < count; index++) {
    const bin = Math.floor(1 + Math.pow(index / count, 2.15) * 168);
    const magnitude = session.bins[bin] / 255;
    const bar = Math.max(2, Math.pow(magnitude, 1.35) * height * .78);
    const top = (height - bar) / 2;
    const barWidth = Math.max(1, stride * .55);
    context2d.globalAlpha = .29 + magnitude * .65;
    context2d.fillRect(index * stride, top, barWidth, bar);
    if (magnitude > .24) {
      context2d.globalAlpha = .92;
      context2d.fillRect(index * stride, top, barWidth, 1);
    }
  }
  context2d.globalAlpha = 1;
}
new ResizeObserver(fitSpectrum).observe(canvas);
reducedMotion.addEventListener("change", () => drawSpectrum(live, true));

function clockLabel(seconds) {
  const round = Math.max(0, Math.floor(seconds));
  return `${Math.floor(round / 60)}:${String(round % 60).padStart(2, "0")}`;
}

function updateTransport(session) {
  const now = session.context.currentTime;
  const current = session.schedule.find(item => now >= item.at && now < item.at + item.duration);
  const paused = session.context.state !== "running";
  session.current = !paused && current;
  if (current && current.segment.seq !== session.lastPlayingSeq) {
    session.lastPlayingSeq = current.segment.seq;
    renderUpcoming(session);
  }
  setText(find("output-heading"), current?.segment.title
    || (session.started ? "Getting the next track ready" : "Getting your mix ready"));
  find("live-progress").hidden = !current;
  if (current) {
    find("playhead").value = Math.min(1, (now - current.at) / current.duration);
    setText(find("elapsed"), clockLabel(now - current.at));
    setText(find("duration"), clockLabel(current.duration));
  }
  pauseButton.disabled = !session.started && !paused;
  setText(pauseButton, paused ? "Resume" : "Pause");
  if (paused) indicator("paused", "Paused");
  else if (current) indicator("playing", "Playing");
  else indicator("preparing", "Preparing");
}

function animate(session, stamp) {
  if (live !== session) return;
  if (stamp - session.lastPaint >= 32) {
    session.lastPaint = stamp;
    updateTransport(session);
    drawSpectrum(session);
  }
  session.frame = requestAnimationFrame(t => animate(session, t));
}

async function acknowledge(session, seq) {
  let error;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      await api(`/api/dj-live/${session.id}/ack`, {seq});
      return;
    } catch (failure) {
      error = failure;
      await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
    }
  }
  throw error;
}

function clearLivePlayer() {
  if (!live) return;
  const session = live;
  live = null;
  clearTimeout(session.timer);
  cancelAnimationFrame(session.frame);
  for (const source of session.sources) {
    try { source.stop(); } catch (_) {}
  }
  session.context.close().catch(() => {});
  syncControls();
  find("live-progress").hidden = true;
  find("next-section").hidden = true;
  find("tracks").replaceChildren();
  setText(find("output-heading"), "Your mix starts here");
  indicator("idle", "Ready");
  drawSpectrum(null);
}

async function scheduleSegment(session, segment) {
  const response = await fetch(segment.audio_url);
  if (!response.ok) throw Error("Could not load the next track.");
  const buffer = await session.context.decodeAudioData(await response.arrayBuffer());
  if (live !== session) return;
  const source = session.context.createBufferSource();
  source.buffer = buffer;
  source.connect(session.analyser);
  const now = session.context.currentTime;
  if (session.started && now > session.nextTime) session.underruns++;
  const at = session.nextTime > now ? session.nextTime : now + .08;
  source.onended = () => {
    session.sources.delete(source);
    session.schedule = session.schedule.filter(item => item.segment.seq !== segment.seq);
    if (live !== session) return;
    session.ackQueue = session.ackQueue.then(() => acknowledge(session, segment.seq)).catch(async () => {
      if (live !== session) return;
      clearLivePlayer();
      activity("Playback stopped because the connection was interrupted. You can start again.", "error");
      try { await api(`/api/dj-live/${session.id}/stop`, {}); } catch (_) {}
    });
  };
  source.start(at);
  session.started = true;
  session.nextTime = at + buffer.duration;
  session.highestScheduled = segment.seq;
  session.sources.add(source);
  session.schedule.push({segment, at, duration: buffer.duration});
  window.dispatchEvent(new CustomEvent("dj-audio-scheduled", {
    detail: {seq: segment.seq, at, duration: buffer.duration, underruns: session.underruns}
  }));
}

const phases = {
  planning: "Choosing what to play next…",
  generating: "Making the next track…",
  listening: "Checking how the next track fits…"
};

// Start on one track when it is long enough to cover making the next one. The next track took
// about 1.2x as long as the first in measured runs, so require 1.75x plus a margin; otherwise
// keep waiting for two tracks.
function canStartEarly(session, segments) {
  if (segments.length < 1) return false;
  if (session.firstReadyAfter === undefined) session.firstReadyAfter = (performance.now() - session.requestedAt) / 1000;
  return segments[0].duration_seconds >= 1.75 * session.firstReadyAfter + 8;
}

async function pollLive(session) {
  if (live !== session) return;
  try {
    const data = await api(`/api/dj-live/${session.id}/status`);
    if (live !== session) return;
    if (data.state === "failed" || data.state === "stopped") {
      clearLivePlayer();
      if (data.state === "failed") {
        console.warn("Live DJ ended:", data.error);
        activity("The mix stopped unexpectedly. You can start again.", "error");
      } else activity("Playback stopped.");
      return;
    }
    session.segments = data.segments;
    renderUpcoming(session);
    if (session.started || data.segments.length >= 2 || data.state === "draining" || canStartEarly(session, data.segments)) {
      for (const segment of data.segments) {
        if (live !== session) return;
        if (segment.seq > session.highestScheduled) await scheduleSegment(session, segment);
      }
    }
    if (live !== session) return;
    if (session.context.state !== "running") activity("Playback is paused. Select Resume to continue.");
    else if (data.state === "draining") activity("Playing what’s ready. The DJ couldn’t make another track.", "error");
    else if (performance.now() < session.steerUntil) return;
    else if (!session.started) activity(phases[data.state] || "Getting your first tracks ready…");
    else if (session.nextTime <= session.context.currentTime + .15) activity("Getting the next track ready…");
    else activity(phases[data.state] || "You can ask for a different sound while the mix plays.");
  } catch (error) {
    if (live === session) activity("Reconnecting to the DJ…", "error");
  } finally {
    if (live === session) session.timer = setTimeout(() => pollLive(session), 500);
  }
}

async function start(request) {
  starting = true;
  syncControls();
  activity("Getting your first tracks ready…");
  indicator("preparing", "Preparing");
  let context;
  try {
    context = new (window.AudioContext || window.webkitAudioContext)();
    await context.resume();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = .76;
    const gain = context.createGain();
    analyser.connect(gain).connect(context.destination);
    const data = await api("/api/dj-live/start", {request});
    const session = {
      id: data.session_id, context, analyser, gain, bins: new Uint8Array(analyser.frequencyBinCount),
      sources: new Set(), schedule: [], segments: [], highestScheduled: -1, nextTime: 0,
      lastPlayingSeq: -1, underruns: 0, ackQueue: Promise.resolve(), lastPaint: -Infinity, steerUntil: 0, requestedAt: performance.now()
    };
    live = session;
    setVolume();
    session.frame = requestAnimationFrame(t => animate(session, t));
    pollLive(session);
  } catch (error) {
    if (context) context.close().catch(() => {});
    activity("Couldn’t start the mix. Try again in a moment.", "error");
    indicator("idle", "Ready");
    console.warn("Live DJ start failed:", error.message);
  } finally {
    starting = false;
    syncControls();
  }
}

async function steer(session, request) {
  steerButton.disabled = true;
  try {
    await api(`/api/dj-live/${session.id}/steer`, {request});
    if (live === session) {
      session.steerUntil = performance.now() + 3000;
      activity("Request sent. You’ll hear the change after the tracks already lined up.", "success");
    }
  } catch (error) {
    if (live === session) activity("Couldn’t send your request. Please try again.", "error");
  } finally {
    steerButton.disabled = false;
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  if (starting || (live && steerButton.disabled)) return;
  const request = input.value.trim();
  if (!request) {
    activity("Tell the DJ what you’d like to hear.", "error");
    input.focus();
    return;
  }
  if (live) steer(live, request);
  else start(request);
});
pauseButton.addEventListener("click", async () => {
  if (!live) return;
  const session = live;
  try {
    if (session.context.state === "running") await session.context.suspend();
    else await session.context.resume();
  } catch (error) {
    if (live === session) activity("Couldn’t change playback. Please try again.", "error");
  }
});
stopButton.addEventListener("click", async () => {
  if (!live) return;
  const {id} = live;
  clearLivePlayer();
  activity("Playback stopped.");
  try { await api(`/api/dj-live/${id}/stop`, {}); }
  catch (error) { if (!live && !starting) activity("Playback stopped. The DJ lost its connection.", "error"); }
});
window.addEventListener("pagehide", () => {
  if (live) navigator.sendBeacon(`/api/dj-live/${live.id}/stop`, new Blob(["{}"], {type: "application/json"}));
});
