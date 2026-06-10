const TARGET_SAMPLE_RATE = 16000;

export function encodeWav(samples, sampleRate = TARGET_SAMPLE_RATE) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

function downsample(buffer, sourceSampleRate, targetSampleRate = TARGET_SAMPLE_RATE) {
  if (sourceSampleRate === targetSampleRate) {
    return buffer;
  }
  const ratio = sourceSampleRate / targetSampleRate;
  const length = Math.round(buffer.length / ratio);
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(Math.floor((index + 1) * ratio), buffer.length);
    let sum = 0;
    let count = 0;
    for (let sourceIndex = start; sourceIndex < end; sourceIndex += 1) {
      sum += buffer[sourceIndex];
      count += 1;
    }
    result[index] = count ? sum / count : 0;
  }
  return result;
}

function mergeBuffers(buffers) {
  const totalLength = buffers.reduce((sum, buffer) => sum + buffer.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const buffer of buffers) {
    merged.set(buffer, offset);
    offset += buffer.length;
  }
  return merged;
}

export async function createPcmRecorder({ onLevel, onChunk, chunkMs = 1000 } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const buffers = [];
  let chunkBuffers = [];
  let chunkTimer = null;

  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const copied = new Float32Array(input);
    buffers.push(copied);
    chunkBuffers.push(copied);
    if (onLevel) {
      const level = Math.sqrt(copied.reduce((sum, value) => sum + value * value, 0) / copied.length);
      onLevel(Math.min(1, level * 8));
    }
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  async function flushChunk() {
    if (!chunkBuffers.length) {
      return;
    }
    const merged = mergeBuffers(chunkBuffers);
    chunkBuffers = [];
    const samples = downsample(merged, audioContext.sampleRate);
    await onChunk(encodeWav(samples));
  }

  if (onChunk) {
    chunkTimer = setInterval(() => {
      flushChunk();
    }, chunkMs);
  }

  return {
    async stop() {
      if (chunkTimer) {
        clearInterval(chunkTimer);
      }
      if (onChunk && chunkBuffers.length) {
        await flushChunk();
      }
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      await audioContext.close();
      const merged = mergeBuffers(buffers);
      const samples = downsample(merged, audioContext.sampleRate);
      return encodeWav(samples);
    },
  };
}
