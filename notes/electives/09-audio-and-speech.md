# EL-09 notes: Audio & Speech

[← Lesson EL-09](../../lessons/electives/09-audio-and-speech.md) · [All notes](../README.md) · [← EL-08 notes](08-gpu-programming.md) · Next: [EL-10 notes →](10-tabular-deep-learning.md)

> **Reading time** ≈ 60 min. **You need:** [DL-04 notes](../deep-learning/04-cnns-computer-vision.md) (CNNs), [DL-06 notes](../deep-learning/06-transformers.md) (encoder-decoder transformers), and [CV-02 notes](../vision/02-vision-transformers-self-supervised.md) §2 (contrastive self-supervision). A little complex-number familiarity helps for §1.

---

## Where we are

Audio is a 1-D signal sampled thousands of times per second. Modern audio ML mostly turns it into one of two things you already know how to handle:

- an **image** (the spectrogram), for CNNs and ViTs;
- a **sequence** (of frames or tokens), for transformers.

The new problems are specific to speech: **alignment** (which audio frames belong to which letters?), and **pretraining on unlabeled audio**.

---

## 1. From waveform to spectrogram

### 1.1 Sampling

A microphone signal is sampled at a rate $f_s$ (16 kHz for speech models, 44.1 or 48 kHz for music). The **Nyquist limit** says frequencies above $f_s/2$ can't be represented. Worse, they **alias**: they fold back and masquerade as lower frequencies.
*Example:* a 9 kHz tone sampled at 16 kHz shows up at $16 - 9 = 7$ kHz. That's why audio is low-pass filtered before downsampling, and why a telephone recording (8 kHz, which cuts everything above 4 kHz) sounds different to a model trained on 16 kHz audio.

### 1.2 The short-time Fourier transform (STFT)

Sound is made of frequencies that change over time. Chop the signal into short overlapping **frames** (for example, 25 ms windows every 10 ms), multiply each frame by a smooth **window** $w$ (to avoid edge artifacts), and take its Fourier transform:

```math
X[m, k] = \sum_{n} x[n]\,w[n - mH]\,e^{-2\pi i kn/N}.
```

$|X[m,k]|^2$ is the **power spectrogram**: energy at frequency bin $k$ in frame $m$. **The time–frequency trade-off:** longer windows give finer frequency resolution but blur timing, and shorter windows the reverse. You can't have both (an uncertainty principle).

### 1.3 Mel and log: matching human hearing

Humans resolve low frequencies finely and high frequencies coarsely. The **mel scale**, $\text{mel}(f) = 2595\log_{10}(1 + f/700)$, is roughly linear below about 1 kHz and logarithmic above.
A **mel filterbank** (for example, 80 triangular filters evenly spaced in mel) pools the STFT bins into perceptually meaningful bands. Then take the **log** (decibels), because loudness perception is roughly logarithmic, and it compresses the huge dynamic range.

**Why log-mel spectrograms instead of raw waveforms?**

- they're compact: 80 numbers per 10 ms, instead of 160 raw samples;
- they discard phase, which barely matters for content;
- they emphasize perceptually relevant structure;
- they're image-like, so CNNs and ViTs apply directly.

Raw-waveform models (wav2vec 2.0's CNN front end) can learn their own filterbank, but they need much more data and compute.

---

## 2. Speech recognition (ASR)

### 2.1 The alignment problem, and CTC

The audio has, say, 500 frames, and the transcript "hello" has 5 characters. Nobody labeled which frames belong to which letter. **Connectionist Temporal Classification (CTC)** sums over **all possible alignments**:

- At each frame, the model outputs a distribution over the characters plus a special **blank** symbol.
- An alignment is a per-frame path, like `h h _ e _ l l _ l o o`. **Collapse it** by merging repeated characters and then removing blanks, which gives `hello`. (The blank between the two `l`s is what allows doubled letters.)
- The probability of the transcript is the sum over every path that collapses to it:

```math
P(y\mid x) = \sum_{\pi\,\in\,\mathcal{B}^{-1}(y)}\ \prod_{t=1}^{T} p_t(\pi_t),\qquad \mathcal{L}_\text{CTC} = -\log P(y\mid x).
```

There are exponentially many paths, but a **dynamic program** (the forward algorithm, as in HMMs) computes the sum in $O(T\cdot\lvert y\rvert)$. The code checks the DP against brute-force enumeration on a tiny example.
CTC assumes the outputs are conditionally independent across frames, so it's often combined with a language model at decode time.

### 2.2 Encoder–decoder ASR

An audio encoder, plus a text decoder with cross-attention (DL-06 §4), generating tokens one at a time. It learns the alignment implicitly through attention, and has an implicit language model in the decoder. **Whisper** is this design.

### 2.3 Self-supervised speech: wav2vec 2.0

Labeled speech is expensive, and unlabeled audio is abundant. wav2vec 2.0:

1. A CNN turns the raw waveform into latent frames (about every 20 ms), and the latents are **quantized** into a learned codebook (discrete "speech units").
2. **Mask** spans of the latent frames, and run a transformer over the sequence.
3. A **contrastive** loss: at each masked position, the transformer's output must pick the **true quantized latent** out of distractors sampled from other positions. (It's InfoNCE again, CV-02 §2.) A diversity loss keeps the codebook in use.

The model learns phonetic structure without any transcripts. Fine-tuning with CTC on as little as **10 minutes to 1 hour** of labeled speech then gives usable ASR.

### 2.4 Whisper: weak supervision at scale

Instead of clean labels or SSL, Whisper trains an encoder-decoder on about **680k hours** of web audio with *existing* (noisy) transcripts. **One model handles many tasks and languages** through special tokens in the decoder's prompt:

- a language token (`<|en|>`, `<|ro|>`);
- a task token (`<|transcribe|>` or `<|translate|>` into English);
- `<|notimestamps|>`, or timestamp tokens.

The decoder learns to condition on them. The scale and diversity make it robust to accents, noise, and domains, but it can **hallucinate** text on silence or noise, a characteristic failure of autoregressive decoders.

### 2.5 Word error rate

Align the hypothesis to the reference with the minimum edit distance, then count **S**ubstitutions, **D**eletions, and **I**nsertions:

```math
\text{WER} = \frac{S + D + I}{N_\text{reference words}} .
```

**It can exceed 100%**, because insertions are unbounded: a model that hallucinates a long sentence over a 2-word reference racks up many insertions. Normalize the text consistently (case, punctuation, numbers) before scoring, or the WER measures formatting, not recognition.

**Debug: great on the test set, poor on your recordings.** Check the mismatches:

- **sampling rate** (8 kHz telephone audio vs 16 kHz, or resampling bugs);
- **channel and microphone**, **noise and reverberation**, far-field audio;
- **accents** and **domain vocabulary** (names, jargon);
- **segmentation** (long silences cause hallucinations, and audio may be cut mid-word);
- **loudness and clipping**;
- **text normalization** in your WER computation.

Fine-tune on a small set of in-domain recordings, augment with noise and reverb, and add a vocabulary prompt or a language-model bias.

---

## 3. Text-to-speech, briefly

**Classic neural pipeline:** text → phonemes (grapheme-to-phoneme) → an **acoustic model** predicts a mel spectrogram (Tacotron-style attention, or FastSpeech-style with explicit durations) → a **vocoder** (WaveNet, HiFi-GAN) turns the mel into a waveform.
**Newer systems** tokenize audio with a **neural codec** (discrete tokens) and generate those tokens with a language model, conditioned on text and a short voice prompt. That's speech generation as next-token prediction (GEN-01).

```python
import numpy as np
from itertools import product
rng = np.random.default_rng(0)

# --- Aliasing: a 9 kHz tone sampled at 16 kHz looks like 7 kHz ------------------------------------------
fs, f_true = 16000, 9000
t = np.arange(fs) / fs
x = np.sin(2 * np.pi * f_true * t)
spectrum = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1 / fs)
print(f"true {f_true} Hz -> apparent peak at {freqs[spectrum.argmax()]:.0f} Hz (Nyquist = {fs//2} Hz)")

# --- STFT by hand: a chirp's dominant frequency rises over time ----------------------------------------------
sig = np.sin(2 * np.pi * (300 + 1500 * t) * t)                       # rising-frequency chirp
win, hop = 400, 160                                                  # 25 ms windows, 10 ms hop
frames = np.stack([sig[i:i + win] * np.hanning(win) for i in range(0, len(sig) - win, hop)])
S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
fbins = np.fft.rfftfreq(win, 1 / fs)
print("dominant frequency in frames 0, 30, 60, 90:", [int(fbins[S[i].argmax()]) for i in (0, 30, 60, 90)], "Hz")
mel = lambda f: 2595 * np.log10(1 + f / 700)
print("mel of 500, 1000, 4000, 8000 Hz:", [round(mel(f)) for f in (500, 1000, 4000, 8000)], "<- compresses high frequencies")
```

```python
# --- CTC: collapse rule, brute force vs dynamic programming -----------------------------------------------------
def collapse(path, blank="_"):
    out, prev = [], None
    for c in path:
        if c != prev and c != blank: out.append(c)
        prev = c
    return "".join(out)
print(collapse("hh_e_ll_lo_"), "|", collapse("hel_lo"), "|", collapse("hello"), "<- 'hello' needs a blank between the l's")

alphabet = ["_", "a", "b"]
T_frames, target = 4, "ab"
probs = rng.dirichlet(np.ones(3), size=T_frames)                      # per-frame distributions
brute = sum(np.prod([probs[t, alphabet.index(c)] for t, c in enumerate(p)])
            for p in product(alphabet, repeat=T_frames) if collapse(p) == target)

def ctc_forward(probs, target):
    ext = ["_"]; [ext.extend([c, "_"]) for c in target]               # _ a _ b _
    S_, T_ = len(ext), len(probs)
    alpha = np.zeros((T_, S_))
    alpha[0, 0] = probs[0, alphabet.index(ext[0])]; alpha[0, 1] = probs[0, alphabet.index(ext[1])]
    for t in range(1, T_):
        for s in range(S_):
            a = alpha[t - 1, s] + (alpha[t - 1, s - 1] if s > 0 else 0)
            if s > 1 and ext[s] != "_" and ext[s] != ext[s - 2]: a += alpha[t - 1, s - 2]
            alpha[t, s] = a * probs[t, alphabet.index(ext[s])]
    return alpha[-1, -1] + alpha[-1, -2]
print(f"P('ab'): brute force over {3**T_frames} paths = {brute:.6f}, forward DP = {ctc_forward(probs, target):.6f}")

# --- Word error rate via edit distance; it can exceed 100% -------------------------------------------------------
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    D = np.zeros((len(r) + 1, len(h) + 1), int); D[:, 0] = range(len(r) + 1); D[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1, j-1] + (r[i-1] != h[j-1]))
    return D[-1, -1] / len(r)
print("WER:", round(wer("the cat sat on the mat", "the cat sat on a mat"), 3),
      "|", round(wer("hello there", "hello there thanks for watching please subscribe"), 2), "<- hallucinated insertions")
```

---

## Pitfalls & misconceptions

- **A sampling-rate mismatch** between training and inference (or a silent resample).
- **Comparing WERs computed with different text normalization.**
- **Feeding long silences to Whisper-style decoders.** Use voice-activity detection or segmentation.
- **Spectrogram "images" with the wrong orientation or scale.** Use log-mel, and normalize per channel.
- **Assuming a test-set WER transfers** to your microphones, accents, and vocabulary.

## Cheat sheet

| Item | Formula / rule |
|---|---|
| Nyquist / aliasing | representable < $f_s/2$; $f > f_s/2$ folds to $f_s - f$ |
| STFT | windowed FFTs of overlapping frames; time vs frequency resolution |
| Mel | $2595\log_{10}(1 + f/700)$; log-mel = mel filterbank + log |
| CTC | $-\log\sum_{\pi\to y}\prod_t p_t(\pi_t)$; collapse repeats, then drop blanks; forward DP |
| wav2vec 2.0 | mask latents → contrastive ID of the quantized target among distractors |
| Whisper | weakly supervised encoder-decoder; task, language, and timestamp tokens |
| WER | $(S+D+I)/N$, can exceed 1 |

## Answer sketches for the lesson's self-check

<details>
<summary>1. Why log-mel spectrograms instead of raw waveforms?</summary>

They're compact and phase-free, perceptually aligned (mel frequency warping, log loudness), less redundant, and image-like, so standard CNNs and ViTs work with far less data than raw-waveform models.
</details>

<details>
<summary>2. What does CTC solve?</summary>

Training with unaligned transcripts. It marginalizes over all frame-level alignments (with a blank symbol and a collapse rule) using an efficient forward DP, so no frame-by-frame labels are needed.
</details>

<details>
<summary>3. What does wav2vec 2.0 learn from unlabeled audio, and how?</summary>

Contextual speech representations that capture phonetic structure. It masks spans of latent frames and trains a transformer to identify the true quantized latent at each masked position among distractors (a contrastive loss), with a codebook diversity loss.
</details>

<details>
<summary>4. How does Whisper handle many tasks and languages in one model?</summary>

Special tokens in the decoder's prompt specify the language, the task (transcribe or translate), and the timestamp mode. The model is trained on 680k hours of diverse, weakly labeled audio covering all these combinations.
</details>

<details>
<summary>5. What does WER count, and why can it exceed 100%?</summary>

Substitutions, deletions, and insertions from the minimum edit alignment, divided by the number of reference words. Insertions are unbounded, so a hallucinated hypothesis can exceed 100% (the demo: 250%).
</details>

<details>
<summary>6. Great on the test set, poor on your recordings.</summary>

Check the sampling rate, microphone and channel, noise and reverberation, accents and vocabulary, segmentation and silence, loudness, and the text normalization in WER. Fix by fine-tuning on in-domain audio, augmentation, and vocabulary prompting.
</details>

## Where this leads

Next: [EL-10 notes](10-tabular-deep-learning.md), the last elective. After images, text, graphs, and audio, it returns to the most common data in industry, tables, and asks when deep learning helps there at all.
