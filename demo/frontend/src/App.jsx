import { ArrowUpDown, FileAudio, Mic, Power, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPcmRecorder, streamAudioFileChunks } from './audio.js';

const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin;
const ENGINE_ORDER_STORAGE_KEY = 'korean-asr-demo.engine-order.v1';

const DEFAULT_DEMO_CONFIG = {
  defaults: {
    language: 'ko',
    beam_size: 1,
    temperature: 0.0,
    mode: 'offline',
    vad: 'silero',
  },
  languages: [{ id: 'ko', label: '한국어' }],
  vad_options: [{ id: 'silero', label: 'Silero' }],
  recording: {
    chunk_ms: 1000,
    sample_rate: 16000,
  },
  streaming: {
    partial_interval_seconds: 1.0,
    min_partial_audio_seconds: 1.0,
    status_poll_interval_ms: 1000,
  },
  ui: {
    mode_change_feedback_ms: 1000,
  },
};

const EMPTY_RESULT = {
  status: '대기',
  text: '',
  error: '',
  data: null,
  metrics: null,
};

export default function App() {
  const [demoConfig, setDemoConfig] = useState(DEFAULT_DEMO_CONFIG);
  const [engines, setEngines] = useState([]);
  const [engineStatuses, setEngineStatuses] = useState({});
  const [selectedModels, setSelectedModels] = useState({});
  const [selectedLanguages, setSelectedLanguages] = useState({});
  const [vadId, setVadId] = useState(DEFAULT_DEMO_CONFIG.defaults.vad);
  const [mode, setMode] = useState(DEFAULT_DEMO_CONFIG.defaults.mode);
  const [results, setResults] = useState({});
  const [status, setStatus] = useState('준비됨');
  const [modeChanged, setModeChanged] = useState(false);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [draggingEngineId, setDraggingEngineId] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isFileProcessing, setIsFileProcessing] = useState(false);
  const [isFileStopping, setIsFileStopping] = useState(false);
  const recorderRef = useRef(null);
  const fileAbortControllerRef = useRef(null);
  const socketsRef = useRef({});
  const draggingEngineIdRef = useRef('');
  const dragPreviewRef = useRef(null);
  const activeRunIdRef = useRef(0);
  const finalTextsRef = useRef({});
  const modeTimerRef = useRef(null);
  const stopWaitTimerRef = useRef(null);
  const chunkAckResolversRef = useRef([]);
  const notifiedEngineErrorsRef = useRef(new Set());

  const options = {
    language: demoConfig.defaults.language,
    beamSize: demoConfig.defaults.beam_size,
    temperature: demoConfig.defaults.temperature,
    chunkMs: demoConfig.recording?.chunk_ms || DEFAULT_DEMO_CONFIG.recording.chunk_ms,
  };

  useEffect(() => {
    async function loadInitialState() {
      try {
        const [configResponse, enginesResponse, statusResponse] = await Promise.all([
          fetch(`${API_BASE}/api/demo-config`),
          fetch(`${API_BASE}/api/engines`),
          fetch(`${API_BASE}/api/engine-status`),
        ]);
        if (!configResponse.ok) {
          throw new Error('데모 설정을 불러오지 못했습니다.');
        }
        if (!enginesResponse.ok) {
          throw new Error('엔진 목록을 불러오지 못했습니다.');
        }
        if (!statusResponse.ok) {
          throw new Error('엔진 상태를 불러오지 못했습니다.');
        }
        const config = await configResponse.json();
        const rows = await enginesResponse.json();
        const statuses = await statusResponse.json();
        const defaultLanguage = config.defaults?.language || DEFAULT_DEMO_CONFIG.defaults.language;

        setDemoConfig(config);
        setMode(config.defaults?.mode || DEFAULT_DEMO_CONFIG.defaults.mode);
        setVadId(config.defaults?.vad || DEFAULT_DEMO_CONFIG.defaults.vad);
        setEngines(applyStoredEngineOrder(rows));
        setEngineStatuses(Object.fromEntries(statuses.map((row) => [row.id, row])));
        setSelectedModels(Object.fromEntries(rows.map((engine) => [engine.id, engine.model])));
        setSelectedLanguages(Object.fromEntries(rows.map((engine) => [engine.id, defaultLanguage])));
        setResults(Object.fromEntries(rows.map((engine) => [engine.id, { ...EMPTY_RESULT }])));
      } catch (exception) {
        showError(exception.message);
      }
    }

    loadInitialState();
  }, []);

  useEffect(() => {
    let active = true;

    async function refreshStatuses() {
      try {
        const rows = await fetchEngineStatuses();
        if (active) {
          setEngineStatuses(Object.fromEntries(rows.map((row) => [row.id, row])));
        }
      } catch {
        // 상태 표시는 보조 기능이라 일시적인 polling 실패는 화면을 막지 않는다.
      }
    }

    refreshStatuses();
    const refreshMs = demoConfig.streaming?.status_poll_interval_ms || DEFAULT_DEMO_CONFIG.streaming.status_poll_interval_ms;
    const timer = window.setInterval(refreshStatuses, refreshMs);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [demoConfig.streaming?.status_poll_interval_ms]);

  useEffect(() => {
    for (const engine of engines) {
      const engineStatus = engineStatuses[engine.id];
      if (engineStatus?.state !== 'error') {
        notifiedEngineErrorsRef.current.delete(engine.id);
        continue;
      }
      if (!isGlobalEngineError(engineStatus)) {
        continue;
      }
      if (notifiedEngineErrorsRef.current.has(engine.id)) {
        continue;
      }
      notifiedEngineErrorsRef.current.add(engine.id);
      showError(`${engine.name}: ${engineStatus.error || engineStatus.label || '엔진 오류가 발생했습니다.'}`);
      break;
    }
  }, [engineStatuses, engines]);

  useEffect(() => () => {
    abortFileProcessing();
    closeStreamingSockets();
    if (modeTimerRef.current) {
      window.clearTimeout(modeTimerRef.current);
    }
    clearStopWaitTimer();
    removeEngineDragPreview();
  }, []);

  const selectedEngineIds = engines.filter((engine) => isEngineActive(engine, engineStatuses[engine.id])).map((engine) => engine.id);
  const selectedEngines = engines.filter((engine) => selectedEngineIds.includes(engine.id));
  const runnableEngines = selectedEngines.filter((engine) => isEngineReady(engineStatuses[engine.id]));
  const gpuCapacity = demoConfig.resources?.gpu_indices?.length || 0;

  function showError(message) {
    setError(String(message || '알 수 없는 오류가 발생했습니다.'));
  }

  function clearError() {
    setError('');
  }

  async function fetchEngineStatuses() {
    const response = await fetch(`${API_BASE}/api/engine-status`);
    if (!response.ok) {
      throw new Error('엔진 상태를 불러오지 못했습니다.');
    }
    return response.json();
  }

  async function refreshEnginesAndStatuses() {
    const [enginesResponse, statuses] = await Promise.all([
      fetch(`${API_BASE}/api/engines`),
      fetchEngineStatuses(),
    ]);
    if (!enginesResponse.ok) {
      throw new Error('엔진 목록을 불러오지 못했습니다.');
    }
    const rows = await enginesResponse.json();
    setEngines((current) => {
      const ordered = mergeEnginesPreservingOrder(current, rows);
      saveEngineOrder(ordered);
      return ordered;
    });
    setEngineStatuses(Object.fromEntries(statuses.map((row) => [row.id, row])));
  }

  async function toggleEngine(engineId) {
    const active = isEngineActive(engineById(engineId, engines), engineStatuses[engineId]);
    const action = active ? 'deactivate' : 'activate';
    updateResult(engineId, { ...EMPTY_RESULT, status: active ? '비활성화 중' : '모델 로딩 중' });
    try {
      const response = await fetch(`${API_BASE}/api/engines/${engineId}/${action}`, { method: 'POST' });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || '모델 상태를 변경하지 못했습니다.');
      }
      await refreshEnginesAndStatuses();
      updateResult(engineId, { ...EMPTY_RESULT });
    } catch (exception) {
      updateResult(engineId, { ...EMPTY_RESULT });
      await refreshEnginesAndStatuses().catch(() => {});
      showError(exception.message);
    }
  }

  function moveEngine(sourceId, targetId) {
    if (!sourceId || !targetId || sourceId === targetId) {
      return;
    }
    setEngines((current) => {
      const sourceIndex = current.findIndex((engine) => engine.id === sourceId);
      const targetIndex = current.findIndex((engine) => engine.id === targetId);
      if (sourceIndex < 0 || targetIndex < 0) {
        return current;
      }
      const next = [...current];
      const [moved] = next.splice(sourceIndex, 1);
      next.splice(targetIndex, 0, moved);
      saveEngineOrder(next);
      return next;
    });
  }

  function removeEngineDragPreview() {
    if (dragPreviewRef.current) {
      dragPreviewRef.current.remove();
      dragPreviewRef.current = null;
    }
  }

  function clearEngineDrag() {
    draggingEngineIdRef.current = '';
    setDraggingEngineId('');
    removeEngineDragPreview();
  }

  function startEngineDrag(event, engine) {
    event.dataTransfer.effectAllowed = 'move';
    event.dataTransfer.setData('text/plain', engine.id);
    draggingEngineIdRef.current = engine.id;
    setDraggingEngineId(engine.id);

    removeEngineDragPreview();
    const preview = document.createElement('div');
    preview.className = 'engine-drag-preview';
    preview.textContent = engine.name;
    document.body.appendChild(preview);
    event.dataTransfer.setDragImage(preview, 18, 18);
    dragPreviewRef.current = preview;
  }

  function enterEngineDropTarget(event, targetId) {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
    moveEngine(draggingEngineIdRef.current, targetId);
  }

  function dropEngine(event, targetId) {
    event.preventDefault();
    const sourceId = event.dataTransfer.getData('text/plain') || draggingEngineIdRef.current;
    moveEngine(sourceId, targetId);
    clearEngineDrag();
  }

  function toggleMode() {
    const nextMode = mode === 'offline' ? 'streaming' : 'offline';
    if (modeTimerRef.current) {
      window.clearTimeout(modeTimerRef.current);
    }
    setMode(nextMode);
    finalTextsRef.current = {};
    setResults(Object.fromEntries(engines.map((engine) => [engine.id, { ...EMPTY_RESULT }])));
    setModeChanged(true);
    modeTimerRef.current = window.setTimeout(() => {
      setModeChanged(false);
      modeTimerRef.current = null;
    }, demoConfig.ui?.mode_change_feedback_ms || DEFAULT_DEMO_CONFIG.ui.mode_change_feedback_ms);
  }

  function updateResult(engineId, patch) {
    setResults((current) => ({
      ...current,
      [engineId]: {
        ...(current[engineId] || EMPTY_RESULT),
        ...patch,
      },
    }));
  }

  function updateResultFrom(engineId, buildPatch) {
    setResults((current) => {
      const previous = current[engineId] || EMPTY_RESULT;
      return {
        ...current,
        [engineId]: {
          ...previous,
          ...buildPatch(previous),
        },
      };
    });
  }

  function updateResultsFor(engineRows, patch) {
    engineRows.forEach((engine) => updateResult(engine.id, patch));
  }

  function resetAllResultsForRun(engineRows, status) {
    const runningEngineIds = new Set(engineRows.map((engine) => engine.id));
    finalTextsRef.current = {};
    setResults(Object.fromEntries(
      engines.map((engine) => [
        engine.id,
        {
          ...EMPTY_RESULT,
          status: runningEngineIds.has(engine.id) ? status : EMPTY_RESULT.status,
        },
      ]),
    ));
  }

  function failActiveRun(message, engineRows = runnableEngines) {
    showError(message);
    setStatus('오류');
    setIsFileProcessing(false);
    setIsFileStopping(false);
    setIsRecording(false);
    fileAbortControllerRef.current = null;
    rejectPendingChunkAcks();
    updateResultsFor(engineRows, { status: '오류', error: message });
  }

  function closeStreamingSockets() {
    for (const socket of Object.values(socketsRef.current)) {
      if ([WebSocket.CONNECTING, WebSocket.OPEN].includes(socket.readyState)) {
        socket.close();
      }
    }
    socketsRef.current = {};
  }

  function cancelVadSocket() {
    const socket = socketsRef.current.vad;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send('cancel');
    }
  }

  function rejectPendingChunkAcks() {
    chunkAckResolversRef.current.forEach((item) => {
      window.clearTimeout(item.timer);
      item.reject(new DOMException('작업이 중지되었습니다.', 'AbortError'));
    });
    chunkAckResolversRef.current = [];
  }

  function abortFileProcessing() {
    if (fileAbortControllerRef.current) {
      fileAbortControllerRef.current.abort();
      fileAbortControllerRef.current = null;
    }
    rejectPendingChunkAcks();
  }

  async function cancelEngineWorkers(engineRows) {
    const requests = engineRows.map(async (engine) => {
      const response = await fetch(`${API_BASE}/api/engines/${engine.id}/cancel-work`, { method: 'POST' });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(`${engine.name}: ${payload.detail || '작업 중지에 실패했습니다.'}`);
      }
    });
    const results = await Promise.allSettled(requests);
    const failed = results.find((result) => result.status === 'rejected');
    if (failed) {
      showError(failed.reason?.message || '일부 엔진 작업을 중지하지 못했습니다.');
    }
    await refreshEnginesAndStatuses().catch(() => {});
  }

  function clearStopWaitTimer() {
    if (stopWaitTimerRef.current) {
      window.clearInterval(stopWaitTimerRef.current);
      stopWaitTimerRef.current = null;
    }
  }

  function beginNewRun() {
    activeRunIdRef.current += 1;
    clearStopWaitTimer();
    abortFileProcessing();
    closeStreamingSockets();
    finalTextsRef.current = {};
    setIsFileStopping(false);
    return activeRunIdRef.current;
  }

  function isActiveRun(runId) {
    return runId === activeRunIdRef.current;
  }

  function validateRunnableEngines() {
    if (!selectedEngines.length) {
      showError('활성화된 엔진이 없습니다. 먼저 사용할 모델을 켜주세요.');
      return false;
    }
    if (selectedEngines.some((engine) => isEnginePreparing(engineStatuses[engine.id]))) {
      showError('모델 초기화가 진행 중입니다. 완료 후 다시 시도해주세요.');
      return false;
    }
    if (!runnableEngines.length) {
      showError('준비 완료된 엔진이 없습니다. 모델 로딩 또는 오류 상태를 확인하세요.');
      return false;
    }
    return true;
  }

  function validateSelectedFile(file) {
    const maxUploadMb = demoConfig.server?.security?.max_upload_mb;
    if (maxUploadMb && file.size > maxUploadMb * 1024 * 1024) {
      showError(`업로드 파일이 너무 큽니다. 최대 ${maxUploadMb}MB까지 허용합니다.`);
      return false;
    }
    return true;
  }

  function selectUploadedFile(file) {
    if (!file) {
      setSelectedFile(null);
      return false;
    }
    if (!validateSelectedFile(file)) {
      setSelectedFile(null);
      return false;
    }
    clearError();
    setSelectedFile(file);
    return true;
  }

  function appendFinalText(engineId, text) {
    const value = String(text || '').trim();
    if (!value) {
      return mergedText(engineId, '');
    }
    const rows = finalTextsRef.current[engineId] || [];
    finalTextsRef.current = {
      ...finalTextsRef.current,
      [engineId]: [...rows, value],
    };
    return mergedText(engineId, '');
  }

  function mergedText(engineId, partialText) {
    const rows = finalTextsRef.current[engineId] || [];
    const partial = String(partialText || '').trim();
    return [...rows, partial].filter(Boolean).join(' ');
  }

  async function transcribeUploadedFile(file) {
    if (!validateRunnableEngines()) {
      return;
    }
    if (!validateSelectedFile(file)) {
      return;
    }

    const runId = beginNewRun();
    const controller = new AbortController();
    fileAbortControllerRef.current = controller;
    setIsFileProcessing(true);
    clearError();
    setLevel(0);
    setStatus('음성 업로드 중');
    resetAllResultsForRun(runnableEngines, '음성 업로드 중');

    try {
      await openVadSocket(runId, '음성 업로드 중', 'file');
      await streamAudioFileChunks({
        file,
        chunkMs: options.chunkMs,
        signal: controller.signal,
        onProgress: (value) => setLevel(value),
        onChunk: async (chunk) => {
          if (!isActiveRun(runId)) {
            return;
          }
          const socket = socketsRef.current.vad;
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send(await chunk.arrayBuffer());
            await waitForChunkAck(runId);
          }
        },
      });
      if (!isActiveRun(runId)) {
        return;
      }
      setStatus('최종 인식 중');
      const socket = socketsRef.current.vad;
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send('stop');
      }
    } catch (exception) {
      if (exception.name === 'AbortError') {
        return;
      }
      if (isActiveRun(runId)) {
        failActiveRun(exception.message);
      }
    }
  }

  async function startRecording() {
    try {
      if (!validateRunnableEngines()) {
        return;
      }

      const runId = beginNewRun();
      clearError();
      setLevel(0);
      setIsRecording(true);
      setStatus(mode === 'streaming' ? '스트리밍 녹음 중' : '녹음 중');
      resetAllResultsForRun(runnableEngines, mode === 'streaming' ? '연결 중' : '대기');

      await openVadSocket(runId, 'VAD 진행 중', 'recording');
      recorderRef.current = await createPcmRecorder({
        chunkMs: options.chunkMs,
        onLevel: setLevel,
        onChunk: async (chunk) => {
          if (!isActiveRun(runId)) {
            return;
          }
          const socket = socketsRef.current.vad;
          if (socket?.readyState === WebSocket.OPEN) {
            socket.send(await chunk.arrayBuffer());
          }
        },
      });
    } catch (exception) {
      closeStreamingSockets();
      failActiveRun(exception.message);
    }
  }

  async function openVadSocket(runId, readyStatus, inputSource) {
    socketsRef.current = {};
    const wsUrl = new URL('/api/vad-stream', API_BASE.replace(/^http/, 'ws'));
    const streamLanguage = selectedLanguages[runnableEngines[0]?.id] || options.language;
    wsUrl.searchParams.set('engine_ids', runnableEngines.map((engine) => engine.id).join(','));
    wsUrl.searchParams.set('vad_id', vadId);
    wsUrl.searchParams.set('stream_mode', mode);
    wsUrl.searchParams.set('language', streamLanguage);
    wsUrl.searchParams.set('beam_size', String(options.beamSize));
    wsUrl.searchParams.set('temperature', String(options.temperature));
    wsUrl.searchParams.set('input_source', inputSource);

    const socket = new WebSocket(wsUrl);
    socket.onmessage = (event) => {
      if (!isActiveRun(runId)) {
        return;
      }
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        failActiveRun('서버 응답을 해석하지 못했습니다.');
        return;
      }
      if (message.type === 'chunk_ack') {
        resolveNextChunkAck();
        return;
      }
      if (message.type === 'status') {
        message.engine_ids.forEach((engineId) => {
          updateResult(engineId, { status: recognitionStatusLabel(message) });
        });
      }
      if (message.type === 'finalizing') {
        updateResultsFor(runnableEngines, { status: '최종 인식 중' });
      }
      if (message.type === 'partial') {
        updateResult(message.engine_id, {
          status: `인식 중 ${message.utterance_index + 1}`,
          data: message.result,
          text: mergedText(message.engine_id, message.result?.text),
        });
      }
      if (message.type === 'utterance_final') {
        const finalText = appendFinalText(message.engine_id, message.result?.text);
        updateResultFrom(message.engine_id, (previous) => ({
          status: Number.isInteger(message.utterance_total) ? `인식 중 (${message.utterance_index + 1}/${message.utterance_total})` : '완료',
          data: message.result,
          text: finalText,
          metrics: addFinalMetrics(previous.metrics, message.result),
        }));
      }
      if (message.type === 'session_final') {
        updateResultsFor(runnableEngines, { status: '완료' });
        socket.close();
        setIsFileProcessing(false);
        setIsFileStopping(false);
        fileAbortControllerRef.current = null;
        setStatus('완료');
      }
      if (message.type === 'error') {
        if (message.engine_id) {
          updateResult(message.engine_id, { status: '오류', error: message.message });
          showError(`${engineNameById(message.engine_id, engines)}: ${message.message}`);
        } else {
          failActiveRun(message.message);
        }
      }
    };
    socket.onerror = () => {
      if (!isActiveRun(runId)) {
        return;
      }
      failActiveRun('서버 연결 중 오류가 발생했습니다.');
    };
    await waitForSocket(socket);
    if (!isActiveRun(runId)) {
      socket.close();
      return;
    }
    socketsRef.current = { vad: socket };
    updateResultsFor(runnableEngines, { status: readyStatus });
  }

  function resolveNextChunkAck() {
    const item = chunkAckResolversRef.current.shift();
    if (!item) {
      return;
    }
    window.clearTimeout(item.timer);
    item.resolve();
  }

  function waitForChunkAck(runId) {
    return new Promise((resolve, reject) => {
      if (!isActiveRun(runId)) {
        reject(new DOMException('작업이 중지되었습니다.', 'AbortError'));
        return;
      }
      const timer = window.setTimeout(() => {
        chunkAckResolversRef.current = chunkAckResolversRef.current.filter((item) => item.reject !== reject);
        reject(new Error('서버 chunk 응답 시간이 초과되었습니다.'));
      }, 10000);
      chunkAckResolversRef.current.push({ resolve, reject, timer });
    });
  }

  async function stopRecording() {
    if (!recorderRef.current) {
      return;
    }
    setIsRecording(false);
    setStatus('최종 인식 중');
    try {
      await recorderRef.current.stop();
      recorderRef.current = null;
      const socket = socketsRef.current.vad;
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send('stop');
      }
    } catch (exception) {
      showError(exception.message);
      setStatus('오류');
    }
  }

  async function submitFile() {
    if (!selectedFile) {
      return;
    }
    await transcribeUploadedFile(selectedFile);
  }

  function stopFileRecognition() {
    const engineRows = [...runnableEngines];
    cancelVadSocket();
    activeRunIdRef.current += 1;
    abortFileProcessing();
    closeStreamingSockets();
    clearStopWaitTimer();
    setIsFileStopping(true);
    setLevel(0);
    setStatus('중지 중');
    updateResultsFor(engineRows, { status: '중지 중' });
    void cancelEngineWorkers(engineRows);
    waitUntilFileDecodingStops(engineRows.map((engine) => engine.id));
  }

  function waitUntilFileDecodingStops(engineIds) {
    stopWaitTimerRef.current = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/engine-status`);
        if (!response.ok) {
          return;
        }
        const rows = await response.json();
        const statuses = Object.fromEntries(rows.map((row) => [row.id, row.state]));
        const busy = engineIds.some((engineId) => ['loading', 'decoding'].includes(statuses[engineId]));
        if (busy) {
          return;
        }
      } catch {
        return;
      }
      clearStopWaitTimer();
      setIsFileProcessing(false);
      setIsFileStopping(false);
      setStatus('완료');
      engineIds.forEach((engineId) => updateResult(engineId, { status: '완료' }));
    }, 600);
  }

  const dropzoneClassName = [
    'dropzone',
    selectedFile ? 'has-file' : '',
    isDraggingFile ? 'dragging' : '',
  ].filter(Boolean).join(' ');
  const modeToggleClassName = [
    'mode-toggle',
    mode === 'streaming' ? 'streaming' : '',
    modeChanged ? 'mode-changed' : '',
  ].filter(Boolean).join(' ');

  return (
    <main className="console-shell">
      <section className="top-console">
        <div className="hero">
          <div>
            <p className="eyebrow">Korean STT Console</p>
            <h1>STT 모델 비교 데모</h1>
          </div>
          <div className="status-block">
            <strong>{selectedEngineIds.length}/{gpuCapacity}</strong>
            <span>동작 모델 / GPU · {status}</span>
          </div>
        </div>

        <div className="upload-panel">
          <label
            className={dropzoneClassName}
            onDragEnter={(event) => {
              event.preventDefault();
              if (isFileProcessing || isFileStopping) {
                return;
              }
              setIsDraggingFile(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              if (isFileProcessing || isFileStopping) {
                return;
              }
              setIsDraggingFile(true);
            }}
            onDragLeave={() => setIsDraggingFile(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDraggingFile(false);
              if (isFileProcessing || isFileStopping) {
                return;
              }
              selectUploadedFile(event.dataTransfer.files?.[0] || null);
            }}
          >
            <FileAudio size={18} />
            <span>{selectedFile ? selectedFile.name : '음성 파일 선택'}</span>
            <input
              type="file"
              accept="audio/*"
              disabled={isFileProcessing || isFileStopping}
              onChange={(event) => {
                if (!selectUploadedFile(event.target.files?.[0] || null)) {
                  event.target.value = '';
                }
              }}
            />
          </label>
          <button
            className={isFileProcessing ? 'secondary-button stop-button' : 'secondary-button'}
            type="button"
            disabled={isFileStopping || ((!selectedFile || !selectedEngineIds.length || isRecording) && !isFileProcessing)}
            onClick={isFileProcessing ? stopFileRecognition : submitFile}
          >
            {isFileStopping ? (
              <span>중지 중</span>
            ) : isFileProcessing ? (
              <>
                <Square size={16} />
                <span>중지</span>
              </>
            ) : '인식'}
          </button>
        </div>

        <div className="control-panel">
          <div className="record-strip">
            <button
              type="button"
              className={isRecording ? 'record-button recording' : 'record-button'}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!selectedEngineIds.length || isFileProcessing || isFileStopping}
            >
              {isRecording ? <Square size={18} /> : <Mic size={18} />}
              <span>{isRecording ? '중지' : '녹음'}</span>
            </button>
            <div className="level">
              <span style={{ width: `${Math.round(level * 100)}%` }} />
            </div>
          </div>

          <div className="mode-strip">
            <button
              type="button"
              className={modeToggleClassName}
              onClick={toggleMode}
              disabled={isRecording || isFileProcessing || isFileStopping}
              aria-pressed={mode === 'streaming'}
            >
              <span className="mode-thumb" />
              <span>Offline</span>
              <span>Streaming</span>
            </button>
          </div>
          <label className="vad-control">
            <span>VAD</span>
            <select value={vadId} onChange={(event) => setVadId(event.target.value)} disabled={isRecording || isFileProcessing || isFileStopping}>
              {demoConfig.vad_options.map((option) => (
                <option key={option.id} value={option.id}>{option.label}</option>
              ))}
            </select>
          </label>
        </div>
      </section>

      {error && <ErrorPopup message={error} onClose={clearError} />}

      <section className="engine-table">
        {engines.map((engine) => (
          <EngineRow
            key={engine.id}
            engine={engine}
            selected={isEngineActive(engine, engineStatuses[engine.id])}
            dragging={draggingEngineId === engine.id}
            result={results[engine.id] || EMPTY_RESULT}
            engineStatus={engineStatuses[engine.id]}
            selectedModel={selectedModels[engine.id] || engine.model}
            selectedLanguage={selectedLanguages[engine.id] || 'ko'}
            modeChanged={modeChanged}
            mode={mode}
            languageOptions={demoConfig.languages}
            onModelChange={(value) => setSelectedModels((current) => ({ ...current, [engine.id]: value }))}
            onLanguageChange={(value) => setSelectedLanguages((current) => ({ ...current, [engine.id]: value }))}
            onToggle={() => toggleEngine(engine.id)}
            toggleDisabled={isRecording || isFileProcessing || isFileStopping}
            onDragStart={(event) => startEngineDrag(event, engine)}
            onDragOver={(event) => {
              event.preventDefault();
              event.dataTransfer.dropEffect = 'move';
            }}
            onDragEnter={(event) => enterEngineDropTarget(event, engine.id)}
            onDrop={(event) => dropEngine(event, engine.id)}
            onDragEnd={clearEngineDrag}
          />
        ))}
      </section>

      <footer className="site-footer">
        <div>
          <span>Developed by Hyungwon Yang</span>
          <a
            href="https://github.com/hyung8758/korean-asr-model-benchmark"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
          >
            <GitHubMark />
          </a>
        </div>
        <p>Mediazen, Inc. All rights reserved.</p>
      </footer>
    </main>
  );
}

function GitHubMark() {
  return (
    <svg className="github-mark" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.38 7.86 10.9.58.1.79-.25.79-.56v-2.16c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.2 1.77 1.2 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.28 1.19-3.08-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.17 1.18A10.93 10.93 0 0 1 12 6.04c.98 0 1.96.13 2.88.39 2.2-1.49 3.17-1.18 3.17-1.18.63 1.58.23 2.75.11 3.04.74.8 1.19 1.83 1.19 3.08 0 4.41-2.69 5.39-5.25 5.67.41.36.78 1.06.78 2.13v3.17c0 .31.21.67.79.56A11.52 11.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}

function ErrorPopup({ message, onClose }) {
  return (
    <div className="error-overlay" role="presentation" onMouseDown={onClose}>
      <section
        className="error-popup"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="error-popup-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div>
          <p id="error-popup-title">오류</p>
          <strong>{message}</strong>
        </div>
        <button type="button" onClick={onClose}>
          확인
        </button>
      </section>
    </div>
  );
}

function EngineRow({ engine, selected, dragging, result, engineStatus, selectedModel, selectedLanguage, modeChanged, mode, languageOptions, onModelChange, onLanguageChange, onToggle, toggleDisabled, onDragStart, onDragOver, onDragEnter, onDrop, onDragEnd }) {
  const metrics = result.metrics;
  const visibleStatus = rowStatus(selected, result, engineStatus, modeChanged, mode);
  const resultText = result.error || result.text || engineStatus?.error || '디코딩 결과 창';
  const hasResultText = Boolean(result.error || result.text || engineStatus?.error);
  const stopRowToggle = (event) => event.stopPropagation();
  return (
    <article
      className={[
        'engine-row',
        engine.theme === 'streaming' ? 'streaming-engine' : '',
        selected ? 'selected' : '',
        dragging ? 'dragging' : '',
      ].filter(Boolean).join(' ')}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDrop={onDrop}
    >
      <div className="engine-selector">
        <span className="text-zone" onClick={stopRowToggle} onKeyDown={stopRowToggle}>
          <div className="engine-title-line">
            <button
              className="drag-handle"
              type="button"
              draggable
              aria-label={`${engine.name} 위치 변경`}
              title="드래그해서 위치 변경"
              onClick={stopRowToggle}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
            >
              <ArrowUpDown size={13} />
            </button>
            <button
              className={`active-toggle ${selected ? 'enabled' : 'disabled'}`}
              type="button"
              aria-pressed={selected}
              aria-label={selected ? `${engine.name} 비활성화` : `${engine.name} 활성화`}
              disabled={toggleDisabled}
              onClick={onToggle}
            >
              <Power size={12} />
            </button>
            <strong>{engine.name}</strong>
          </div>
          <div className="engine-controls">
            <div>
              <select
                className="model-select"
                value={selectedModel}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
                onChange={(event) => onModelChange(event.target.value)}
              >
                {modelOptionsFor(engine).map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </div>
            <div>
              <select
                className="language-select"
                value={selectedLanguage}
                onClick={(event) => event.stopPropagation()}
                onKeyDown={(event) => event.stopPropagation()}
                onChange={(event) => onLanguageChange(event.target.value)}
              >
                {languageOptions.map((option) => (
                  <option key={option.id} value={option.id}>{option.label}</option>
                ))}
              </select>
            </div>
          </div>
        </span>
      </div>
      <div className="engine-result">
        <div className="decode-line text-zone" onClick={stopRowToggle} onKeyDown={stopRowToggle}>
          <span className={[
            visibleStatus.active ? 'motion-label' : '',
            visibleStatus.modeChanging ? 'mode-status-label' : '',
          ].filter(Boolean).join(' ')}>{visibleStatus.label}</span>
          <p className={hasResultText ? 'result-text' : 'result-placeholder'}>{resultText}</p>
        </div>
        <div className="mini-metrics text-zone" onClick={stopRowToggle} onKeyDown={stopRowToggle}>
          <MiniMetric label="audio duration" value={formatMetricPair(metrics, 'audioDuration', formatSeconds)} />
          <MiniMetric label="decode time" value={formatMetricPair(metrics, 'decodeTime', formatSeconds)} tone="blue" />
          <MiniMetric label="total time" value={formatMetricPair(metrics, 'totalTime', formatSeconds)} tone="green" />
          <MiniMetric label="RTF" value={formatMetricPair(metrics, 'rtf', formatNumber)} tone="violet" />
        </div>
      </div>
    </article>
  );
}

function modelSizeLabel(model) {
  const text = String(model || '');
  const knownSizes = ['large-v3'];
  const matched = knownSizes.find((size) => text.toLowerCase().includes(size));
  return matched || text.split('/').pop() || text;
}

function modelOptionsFor(engine) {
  const options = engine.model_options?.length ? engine.model_options : [engine.model];
  return options.map((model) => ({ value: model, label: modelSizeLabel(model) }));
}

function engineById(engineId, engines) {
  return engines.find((engine) => engine.id === engineId) || null;
}

function mergeEnginesPreservingOrder(currentEngines, nextEngines) {
  if (!currentEngines.length) {
    return applyStoredEngineOrder(nextEngines);
  }
  const nextById = new Map(nextEngines.map((engine) => [engine.id, engine]));
  const ordered = currentEngines
    .map((engine) => nextById.get(engine.id))
    .filter(Boolean);
  const knownIds = new Set(ordered.map((engine) => engine.id));
  const added = nextEngines.filter((engine) => !knownIds.has(engine.id));
  return [...ordered, ...added];
}

function applyStoredEngineOrder(engineRows) {
  return orderEnginesByIds(engineRows, readStoredEngineOrder());
}

function orderEnginesByIds(engineRows, engineIds) {
  if (!engineIds.length) {
    return engineRows;
  }
  const engineByIdMap = new Map(engineRows.map((engine) => [engine.id, engine]));
  const ordered = engineIds.map((engineId) => engineByIdMap.get(engineId)).filter(Boolean);
  const orderedIds = new Set(ordered.map((engine) => engine.id));
  const added = engineRows.filter((engine) => !orderedIds.has(engine.id));
  return [...ordered, ...added];
}

function readStoredEngineOrder() {
  try {
    const value = window.localStorage.getItem(ENGINE_ORDER_STORAGE_KEY);
    const parsed = value ? JSON.parse(value) : [];
    return Array.isArray(parsed) ? parsed.filter((item) => typeof item === 'string') : [];
  } catch {
    return [];
  }
}

function saveEngineOrder(engineRows) {
  try {
    window.localStorage.setItem(ENGINE_ORDER_STORAGE_KEY, JSON.stringify(engineRows.map((engine) => engine.id)));
  } catch {
  }
}

function isEngineActive(engine, engineStatus) {
  if (!engine) {
    return false;
  }
  if (engineStatus?.state) {
    return engineStatus.state !== 'inactive';
  }
  return Boolean(engine.active);
}

function isEngineReady(engineStatus) {
  return engineStatus?.state === 'ready';
}

function isEnginePreparing(engineStatus) {
  return ['loading', 'not_loaded', 'unloading'].includes(engineStatus?.state);
}

function engineNameById(engineId, engines) {
  return engines.find((engine) => engine.id === engineId)?.name || engineId;
}

function recognitionStatusLabel(message) {
  if (message.status !== '인식 중') {
    return message.status;
  }
  if (Number.isInteger(message.utterance_index) && Number.isInteger(message.utterance_total)) {
    return `인식 중 (${message.utterance_index + 1}/${message.utterance_total})`;
  }
  if (Number.isInteger(message.utterance_index)) {
    return `인식 중 ${message.utterance_index + 1}`;
  }
  return message.status;
}

function isActiveStatus(status) {
  return ['모델 로딩 중', 'server 시작 중', '비활성화 중', '인식 중', '연결 중', '스트리밍 중', '최종 인식 중', 'VAD 진행 중'].some((word) => status.startsWith(word));
}

function rowStatus(selected, result, engineStatus, modeChanged, mode) {
  if (!selected || engineStatus?.state === 'inactive') {
    return { label: '비활성화', active: false };
  }
  if (modeChanged) {
    return { label: mode === 'streaming' ? 'streaming 모드' : 'offline 모드', active: false, modeChanging: true };
  }
  if (result.status && !['대기', '완료'].includes(result.status)) {
    return { label: result.status, active: isActiveStatus(result.status) };
  }
  if (engineStatus?.state === 'loading') {
    return { label: '모델 로딩 중', active: true };
  }
  if (isGlobalEngineError(engineStatus)) {
    return { label: engineStatus.label || '로딩 실패', active: false };
  }
  if (result.status === '완료') {
    return { label: '완료', active: false };
  }
  const idleLabel = ['decoding', 'error'].includes(engineStatus?.state) ? '준비 완료' : engineStatus?.label || '준비 완료';
  return { label: idleLabel, active: false };
}

function isGlobalEngineError(engineStatus) {
  return engineStatus?.state === 'error' && ['로딩 실패', 'server 미실행'].includes(engineStatus.label);
}

function MiniMetric({ label, value, tone = 'gray' }) {
  return (
    <div className={`mini-metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function formatSeconds(value) {
  if (!isFiniteNumber(value)) {
    return '-';
  }
  return `${Number(value).toFixed(2)}s`;
}

function formatNumber(value) {
  if (!isFiniteNumber(value)) {
    return '-';
  }
  return Number(value).toFixed(2);
}

function formatMetricPair(metrics, key, formatter) {
  const average = metricAverage(metrics, key);
  const last = metrics?.[key]?.last;
  if (!isFiniteNumber(average) && !isFiniteNumber(last)) {
    return '-';
  }
  return (
    <span className="metric-pair">
      <span>avg {formatter(average)}</span>
      <span>last {formatter(last)}</span>
    </span>
  );
}

function addFinalMetrics(previousMetrics, data) {
  const metrics = previousMetrics || createMetricStats();
  const next = {
    ...metrics,
    audioDuration: addMetricValue(metrics.audioDuration, data?.audio_duration),
    decodeTime: addMetricValue(metrics.decodeTime, data?.decode_time),
    totalTime: addMetricValue(metrics.totalTime, data?.total_time),
    rtf: addMetricValue(metrics.rtf, data?.rtf),
  };

  const audioDuration = numberOrNull(data?.audio_duration);
  const decodeTime = numberOrNull(data?.decode_time);
  if (audioDuration !== null) {
    next.totalAudioDuration += audioDuration;
  }
  if (decodeTime !== null) {
    next.totalDecodeTime += decodeTime;
  }
  return next;
}

function createMetricStats() {
  return {
    totalAudioDuration: 0,
    totalDecodeTime: 0,
    audioDuration: createMetricBucket(),
    decodeTime: createMetricBucket(),
    totalTime: createMetricBucket(),
    rtf: createMetricBucket(),
  };
}

function createMetricBucket() {
  return { sum: 0, count: 0, last: null };
}

function addMetricValue(bucket, value) {
  const number = numberOrNull(value);
  if (number === null) {
    return bucket;
  }
  return {
    sum: bucket.sum + number,
    count: bucket.count + 1,
    last: number,
  };
}

function metricAverage(metrics, key) {
  if (!metrics) {
    return null;
  }
  if (key === 'rtf' && metrics.totalAudioDuration > 0) {
    return metrics.totalDecodeTime / metrics.totalAudioDuration;
  }
  const bucket = metrics[key];
  if (!bucket?.count) {
    return null;
  }
  return bucket.sum / bucket.count;
}

function numberOrNull(value) {
  if (value === undefined || value === null || value === '') {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function isFiniteNumber(value) {
  return numberOrNull(value) !== null;
}

function waitForSocket(socket) {
  return new Promise((resolve, reject) => {
    if (socket.readyState === WebSocket.OPEN) {
      resolve();
      return;
    }
    const cleanup = () => {
      socket.removeEventListener('open', handleOpen);
      socket.removeEventListener('error', handleError);
    };
    const handleOpen = () => {
      cleanup();
      resolve();
    };
    const handleError = () => {
      cleanup();
      reject(new Error('WebSocket 연결에 실패했습니다.'));
    };
    socket.addEventListener('open', handleOpen);
    socket.addEventListener('error', handleError);
  });
}
