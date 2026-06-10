import { FileAudio, Mic, Square } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPcmRecorder } from './audio.js';

const API_BASE = import.meta.env.VITE_API_BASE || `${window.location.protocol}//${window.location.hostname}:16000`;

const DEFAULT_OPTIONS = {
  language: 'ko',
  beamSize: 1,
  temperature: 0.0,
  chunkMs: 1000,
};

const EMPTY_RESULT = {
  status: '대기',
  text: '',
  error: '',
  data: null,
};

export default function App() {
  const [engines, setEngines] = useState([]);
  const [engineStatuses, setEngineStatuses] = useState({});
  const [selectedEngineIds, setSelectedEngineIds] = useState([]);
  const [mode, setMode] = useState('offline');
  const options = DEFAULT_OPTIONS;
  const [results, setResults] = useState({});
  const [status, setStatus] = useState('준비됨');
  const [level, setLevel] = useState(0);
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const recorderRef = useRef(null);
  const socketsRef = useRef({});
  const abortControllersRef = useRef([]);
  const activeRunIdRef = useRef(0);
  const completedStreamEnginesRef = useRef(new Set());
  const expectedStreamEnginesRef = useRef(0);

  useEffect(() => {
    fetch(`${API_BASE}/api/engines`)
      .then((response) => response.json())
      .then((rows) => {
        setEngines(rows);
        setSelectedEngineIds(rows.map((engine) => engine.id));
        setResults(Object.fromEntries(rows.map((engine) => [engine.id, { ...EMPTY_RESULT }])));
      })
      .catch((exception) => setError(`엔진 목록을 불러오지 못했습니다: ${exception.message}`));
  }, []);

  useEffect(() => {
    let active = true;

    async function refreshStatuses() {
      try {
        const response = await fetch(`${API_BASE}/api/engine-status`);
        if (!response.ok) {
          return;
        }
        const rows = await response.json();
        if (active) {
          setEngineStatuses(Object.fromEntries(rows.map((row) => [row.id, row])));
        }
      } catch {
        // 상태 표시는 보조 기능이라 일시적인 polling 실패는 화면을 막지 않는다.
      }
    }

    refreshStatuses();
    const timer = window.setInterval(refreshStatuses, 1000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => () => {
    abortPendingRequests();
    closeStreamingSockets();
  }, []);

  const selectedEngines = engines.filter((engine) => selectedEngineIds.includes(engine.id));
  const runnableEngines = selectedEngines.filter((engine) => engineStatuses[engine.id]?.state !== 'error');

  function toggleEngine(engineId) {
    setSelectedEngineIds((current) => (
      current.includes(engineId) ? current.filter((id) => id !== engineId) : [...current, engineId]
    ));
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

  function closeStreamingSockets() {
    for (const socket of Object.values(socketsRef.current)) {
      if ([WebSocket.CONNECTING, WebSocket.OPEN].includes(socket.readyState)) {
        socket.close();
      }
    }
    socketsRef.current = {};
  }

  function abortPendingRequests() {
    abortControllersRef.current.forEach((controller) => controller.abort());
    abortControllersRef.current = [];
  }

  function beginNewRun() {
    activeRunIdRef.current += 1;
    abortPendingRequests();
    closeStreamingSockets();
    completedStreamEnginesRef.current = new Set();
    expectedStreamEnginesRef.current = 0;
    return activeRunIdRef.current;
  }

  function isActiveRun(runId) {
    return runId === activeRunIdRef.current;
  }

  function markStreamEngineDone(engineId, runId) {
    if (!isActiveRun(runId)) {
      return;
    }
    const completed = completedStreamEnginesRef.current;
    if (completed.has(engineId)) {
      return;
    }
    completed.add(engineId);
    if (completed.size >= expectedStreamEnginesRef.current) {
      setStatus('완료');
    }
  }

  function validateRunnableEngines() {
    if (!selectedEngines.length) {
      setError('최소 하나 이상의 엔진을 선택해야 합니다.');
      return false;
    }
    if (!runnableEngines.length) {
      setError('실행 가능한 엔진이 없습니다. 로딩 실패 또는 server 미실행 상태를 확인하세요.');
      return false;
    }
    return true;
  }

  async function transcribeOne(blob, filename, engine, runId) {
    updateResult(engine.id, { status: '인식 중', error: '', text: '' });
    const controller = new AbortController();
    abortControllersRef.current.push(controller);
    const formData = new FormData();
    formData.append('file', blob, filename);
    formData.append('engine_id', engine.id);
    formData.append('language', options.language);
    formData.append('beam_size', String(options.beamSize));
    formData.append('temperature', String(options.temperature));

    try {
      const response = await fetch(`${API_BASE}/api/transcribe`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      if (!isActiveRun(runId)) {
        return;
      }
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || response.statusText);
      }
      const data = await response.json();
      if (!isActiveRun(runId)) {
        return;
      }
      updateResult(engine.id, { status: '완료', data, text: data.text || '' });
    } finally {
      abortControllersRef.current = abortControllersRef.current.filter((item) => item !== controller);
    }
  }

  async function transcribeSelected(blob, filename = 'recording.wav') {
    if (!validateRunnableEngines()) {
      return;
    }

    const runId = beginNewRun();
    setStatus('디코딩 중');
    setError('');
    selectedEngines
      .filter((engine) => engineStatuses[engine.id]?.state === 'error')
      .forEach((engine) => updateResult(engine.id, {
        status: engineStatuses[engine.id]?.label || '오류',
        error: engineStatuses[engine.id]?.error || '실행할 수 없는 엔진입니다.',
      }));
    await Promise.all(runnableEngines.map(async (engine) => {
      try {
        await transcribeOne(blob, filename, engine, runId);
      } catch (exception) {
        if (!isActiveRun(runId) || exception.name === 'AbortError') {
          return;
        }
        updateResult(engine.id, { status: '오류', error: exception.message, data: null, text: '' });
      }
    }));
    if (isActiveRun(runId)) {
      setStatus('완료');
    }
  }

  async function startRecording() {
    try {
      if (!validateRunnableEngines()) {
        return;
      }

      const runId = beginNewRun();
      setError('');
      setLevel(0);
      setIsRecording(true);
      setStatus(mode === 'streaming' ? '스트리밍 녹음 중' : '녹음 중');
      runnableEngines.forEach((engine) => updateResult(engine.id, { ...EMPTY_RESULT, status: mode === 'streaming' ? '연결 중' : '대기' }));

      if (mode === 'streaming') {
        await openStreamingSockets(runId);
        recorderRef.current = await createPcmRecorder({
          chunkMs: options.chunkMs,
          onLevel: setLevel,
          onChunk: async (chunk) => {
            if (!isActiveRun(runId)) {
              return;
            }
            const buffer = await chunk.arrayBuffer();
            for (const socket of Object.values(socketsRef.current)) {
              if (socket.readyState === WebSocket.OPEN) {
                socket.send(buffer.slice(0));
              }
            }
          },
        });
        return;
      }

      recorderRef.current = await createPcmRecorder({ onLevel: setLevel });
    } catch (exception) {
      closeStreamingSockets();
      setIsRecording(false);
      setStatus('오류');
      setError(exception.message);
    }
  }

  async function openStreamingSockets(runId) {
    socketsRef.current = {};
    completedStreamEnginesRef.current = new Set();
    expectedStreamEnginesRef.current = runnableEngines.length;
    for (const engine of runnableEngines) {
      const wsUrl = new URL('/api/stream', API_BASE.replace(/^http/, 'ws'));
      wsUrl.searchParams.set('engine_id', engine.id);
      wsUrl.searchParams.set('language', options.language);
      wsUrl.searchParams.set('beam_size', String(options.beamSize));
      wsUrl.searchParams.set('temperature', String(options.temperature));

      const socket = new WebSocket(wsUrl);
      socket.onmessage = (event) => {
        if (!isActiveRun(runId)) {
          return;
        }
        const message = JSON.parse(event.data);
        if (message.type === 'finalizing') {
          updateResult(engine.id, { status: '최종 인식 중' });
        }
        if (message.type === 'partial') {
          updateResult(engine.id, {
            status: `인식 중 ${message.chunk_index}`,
            data: message.result,
            text: message.result?.text || '',
          });
        }
        if (message.type === 'final') {
          updateResult(engine.id, {
            status: '완료',
            data: message.result,
            text: message.result?.text || '',
          });
          socket.close();
          markStreamEngineDone(engine.id, runId);
        }
        if (message.type === 'error') {
          updateResult(engine.id, { status: '오류', error: message.message });
          socket.close();
          markStreamEngineDone(engine.id, runId);
        }
      };
      await waitForSocket(socket);
      if (!isActiveRun(runId)) {
        socket.close();
        return;
      }
      socketsRef.current[engine.id] = socket;
      updateResult(engine.id, { status: '스트리밍 중' });
    }
  }

  async function stopRecording() {
    if (!recorderRef.current) {
      return;
    }
    setIsRecording(false);
    setStatus(mode === 'streaming' ? '최종 인식 중' : '디코딩 중');
    try {
      const blob = await recorderRef.current.stop();
      recorderRef.current = null;
      if (mode === 'streaming') {
        for (const socket of Object.values(socketsRef.current)) {
          if (socket.readyState === WebSocket.OPEN) {
            socket.send('stop');
          }
        }
      } else {
        await transcribeSelected(blob);
      }
    } catch (exception) {
      setError(exception.message);
      setStatus('오류');
    }
  }

  async function submitFile() {
    if (!selectedFile) {
      return;
    }
    await transcribeSelected(selectedFile, selectedFile.name);
  }

  const dropzoneClassName = [
    'dropzone',
    selectedFile ? 'has-file' : '',
    isDraggingFile ? 'dragging' : '',
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
            <strong>{selectedEngineIds.length}/{engines.length}</strong>
            <span>{status}</span>
          </div>
        </div>

        <div className="upload-panel">
          <label
            className={dropzoneClassName}
            onDragEnter={(event) => {
              event.preventDefault();
              setIsDraggingFile(true);
            }}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDraggingFile(true);
            }}
            onDragLeave={() => setIsDraggingFile(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDraggingFile(false);
              setSelectedFile(event.dataTransfer.files?.[0] || null);
            }}
          >
            <FileAudio size={18} />
            <span>{selectedFile ? selectedFile.name : '음성 파일 선택'}</span>
            <input type="file" accept="audio/*" onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} />
          </label>
          <button className="secondary-button" type="button" disabled={!selectedFile || !selectedEngineIds.length || isRecording} onClick={submitFile}>
            인식
          </button>
        </div>

        <div className="control-panel">
          <div className="record-strip">
            <button
              type="button"
              className={isRecording ? 'record-button recording' : 'record-button'}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!selectedEngineIds.length}
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
              className={mode === 'streaming' ? 'mode-toggle streaming' : 'mode-toggle'}
              onClick={() => setMode(mode === 'offline' ? 'streaming' : 'offline')}
              disabled={isRecording}
              aria-pressed={mode === 'streaming'}
            >
              <span className="mode-thumb" />
              <span>Offline</span>
              <span>Streaming</span>
            </button>
          </div>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="engine-table">
        {engines.map((engine) => (
          <EngineRow
            key={engine.id}
            engine={engine}
            selected={selectedEngineIds.includes(engine.id)}
            result={results[engine.id] || EMPTY_RESULT}
            engineStatus={engineStatuses[engine.id]}
            onToggle={() => toggleEngine(engine.id)}
          />
        ))}
      </section>
    </main>
  );
}

function EngineRow({ engine, selected, result, engineStatus, onToggle }) {
  const data = result.data;
  const visibleStatus = rowStatus(selected, result, engineStatus);
  const resultText = result.error || result.text || engineStatus?.error || '디코딩 결과 창';
  const hasResultText = Boolean(result.error || result.text || engineStatus?.error);
  return (
    <article
      className={selected ? 'engine-row selected' : 'engine-row'}
      role="button"
      tabIndex={0}
      onClick={onToggle}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onToggle();
        }
      }}
    >
      <div className="engine-selector">
        <span>
          <strong>{engine.name}</strong>
          <small>model size: {modelSizeLabel(engine.model)}</small>
        </span>
      </div>
      <div className="engine-result">
        <div className="decode-line">
          <span className={visibleStatus.active ? 'motion-label' : ''}>{visibleStatus.label}</span>
          <p className={hasResultText ? 'result-text' : 'result-placeholder'}>{resultText}</p>
        </div>
        <div className="mini-metrics">
          <MiniMetric label="audio duration" value={formatSeconds(data?.audio_duration)} />
          <MiniMetric label="decode time" value={formatSeconds(data?.decode_time)} tone="blue" />
          <MiniMetric label="total time" value={formatSeconds(data?.total_time)} tone="green" />
          <MiniMetric label="RTF" value={data?.rtf ?? '-'} tone="violet" />
        </div>
      </div>
    </article>
  );
}

function modelSizeLabel(model) {
  const text = String(model || '');
  const knownSizes = ['large-v3-turbo', 'large-v3', 'large', 'medium', 'small', 'base', 'tiny'];
  const matched = knownSizes.find((size) => text.toLowerCase().includes(size));
  return matched || text.split('/').pop() || text;
}

function rowStatus(selected, result, engineStatus) {
  if (!selected) {
    return { label: '비선택', active: false };
  }
  if (result.status && !['대기', '완료'].includes(result.status)) {
    return { label: result.status, active: ['인식 중', '연결 중', '스트리밍 중', '최종 인식 중'].some((word) => result.status.startsWith(word)) };
  }
  if (engineStatus?.state === 'loading') {
    return { label: '모델 로딩 중', active: true };
  }
  if (engineStatus?.state === 'decoding') {
    return { label: '인식 중', active: true };
  }
  if (engineStatus?.state === 'error') {
    return { label: engineStatus.label || '로딩 실패', active: false };
  }
  if (result.status === '완료') {
    return { label: '완료', active: false };
  }
  return { label: engineStatus?.label || '준비 완료', active: false };
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
  if (value === undefined || value === null) {
    return '-';
  }
  return `${Number(value).toFixed(3)}s`;
}

function waitForSocket(socket) {
  return new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = () => reject(new Error('WebSocket 연결에 실패했습니다.'));
  });
}
