# EL-09: Audio & Speech

| Track | Time | Level | Prerequisites |
|---|---|---|---|
| Elective | ~8 h | L2 | DL-04, DL-06 |

## Why this matters
Speech recognition, voice assistants, audio classification, music, and text-to-speech all rely on the same deep-learning toolkit, applied to a
signal that needs its own preprocessing: sampling, spectrograms, and mel features. Modern speech models (wav2vec 2.0, Whisper) are transformers
trained with self-supervision or weak supervision, so this elective reuses a lot of what you already know.

## Learning goals
By the end you can:
- Explain sampling rate, the STFT, spectrograms, and mel/log-mel features.
- Build an audio classifier on spectrograms with a CNN or a pretrained audio transformer.
- Explain CTC and encoder-decoder ASR, self-supervised speech pretraining (wav2vec 2.0), and Whisper's weakly supervised approach.
- Fine-tune a speech-recognition model, and evaluate it with word error rate (WER).
- Describe text-to-speech pipelines at a high level.

## Study plan

| # | Step | Resource | Scope | Time |
|---|---|---|---|---|
| 1 | **Read + Build** | [Hugging Face Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) | Unit 1 (working with audio data), Unit 2 (audio applications with pipelines), Unit 3 (transformer architectures for audio) | 3 h |
| 2 | **Read + Build** | [HF Audio Course](https://huggingface.co/learn/audio-course/chapter0/introduction) | The audio-classification and speech-recognition units (fine-tune a model and compute WER) | 3 h |
| 3 | **Read** | [Whisper](https://arxiv.org/abs/2212.04356) | §1–3 (data, model, multitask format) | 1 h |
| 4 | *Read (optional)* | [Jurafsky & Martin, SLP3](https://web.stanford.edu/~jurafsky/slp3/) | The speech chapters (ASR and TTS) | 1 h |

## Check your understanding
1. Why do we usually work with log-mel spectrograms rather than raw waveforms?
2. What problem does CTC loss solve for speech recognition?
3. What does wav2vec 2.0 learn from unlabeled audio, and how?
4. How does Whisper handle multiple tasks and languages in one model?
5. What does WER count, and why can it exceed 100%?
6. *(debug)* Your fine-tuned ASR model is great on the test set but poor on your own recordings. What mismatches do you check?

## Mini-project
**Task:** Fine-tune a small Whisper model on a low-resource language or an accented-speech subset, and report WER before and after.
Separately, train a spectrogram CNN classifier on an environmental-sound dataset.
**Deliverable:** A WER table, a confusion matrix, and 5 transcription error examples.

## Go deeper
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) · [WaveNet](https://arxiv.org/abs/1609.03499)
- [Whisper model docs](https://github.com/huggingface/transformers): the Whisper model page in the `transformers` docs.

## Toolbox, papers & practice
- **Reference shelf:** [Toolbox 12: Specialized topics](../../toolbox/12-specialized-topics.md) (audio & speech).
- **Papers:** [wav2vec 2.0](https://arxiv.org/abs/2006.11477) · [Whisper](https://arxiv.org/abs/2212.04356) · [WaveNet](https://arxiv.org/abs/1609.03499).
- **Implement it yourself:** STFT and a mel filterbank in NumPy. Check them against `librosa`/`torchaudio` output.
- **Drills:** more in [exercises/](../../exercises/README.md).
