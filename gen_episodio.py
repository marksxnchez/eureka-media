"""Genera la voz de un episodio con el clon de Marc (versión de ayer: final_lora_adapter.pt), limpia.

Uso en Colab (dentro de /content/cbft, con Drive montado y audio_data descomprimido):
    python gen_episodio.py 27 "¿Sabías que...?" "Frase 2" ...
    o bien: python gen_episodio.py 27 --guion guion.txt

Por cada frase:
  - hasta 6 tomas; se queda la que Whisper (medium) entiende mejor (para al pasar de 0.98)
  - si "branquias" u otras palabras difíciles fallan, prueba variantes de escritura
  - descarta tomas con recortes (clipping), silencios largos dentro o duración rara
  - limpieza: quita silencios de los bordes, filtro de graves (70 Hz), fundidos de 15 ms,
    mismo volumen en todas las frases, margen de 0,15 s antes y 0,35 s después (no se corta ninguna palabra)
Salida en Drive: Eureka clon voz/episodios/<num>/ (frases sueltas + voz.wav + informe.txt)
"""
import sys, os, re, numpy as np, torch, soundfile as sf, pandas as pd, librosa
from scipy.signal import butter, sosfiltfilt
from difflib import SequenceMatcher
sys.path.insert(0, ".")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from lora import inject_lora_layers
from faster_whisper import WhisperModel

D = "/content/drive/MyDrive/Eureka clon voz"
ADAPTER = os.environ.get("ADAPTER", D + "/final_lora_adapter.pt")
num = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "--guion":
    T = [l.strip() for l in open(sys.argv[3]) if l.strip() and not l.startswith("#") and not re.match(r"(?i)\s*pill?ar\s*:", l)]
else:
    T = sys.argv[2:]
OUT = f"{D}/episodios/{num}"; os.makedirs(OUT, exist_ok=True)

ck = torch.load(ADAPTER, map_location="cuda")
cfg = ck["lora_config"]
m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
L = inject_lora_layers(m.t3.tfmr, cfg["target_modules"], rank=cfg["rank"], alpha=cfg["alpha"], dropout=0.0)
if "lora_weights" in ck:
    W = {k: (v["lora_A"], v["lora_B"]) for k, v in ck["lora_weights"].items()}
else:
    s = ck["lora_state_dict"]; W = {k[:-7]: (s[k], s[k[:-7] + ".lora_B"]) for k in s if k.endswith(".lora_A")}
n = 0
for k, (a, b) in W.items():
    if k in L: L[k].lora_A.data = a.to("cuda"); L[k].lora_B.data = b.to("cuda"); n += 1
print("capas LoRA cargadas:", n, flush=True)

df = pd.read_csv("audio_data/metadata.csv")
qs = df[df.transcription.str.strip().str.endswith("?") & (df.duration_seconds > 2.5) & (df.duration_seconds < 8)]
REFQ = "audio_data/" + qs.iloc[0].file_name
REFA = "audio_data/" + df[(df.duration_seconds > 7) & (df.duration_seconds < 10)].iloc[0].file_name
asr = WhisperModel("medium", device="cuda", compute_type="float16")
SR = m.sr
norm = lambda t: re.sub(r"[^a-záéíóúñü]", "", t.lower())
VARIANTES = {"branquias": ["branquias", "bránquias", "brankias"], "pulpo": ["pulpo", "púlpo"]}
HP = butter(4, 70, btype="highpass", fs=SR, output="sos")

def variantes(t):
    out = [t]
    for w, alts in VARIANTES.items():
        if w in t.lower():
            out += [re.sub(w, a, t, flags=re.I) for a in alts[1:]]
    return out

def defectos(w):
    if np.max(np.abs(w)) > 0.995: return "recorte"
    fr = librosa.feature.rms(y=w, frame_length=1024, hop_length=256)[0]
    act = fr > 0.02 * fr.max()
    idx = np.where(act)[0]
    if len(idx) < 2: return "vacía"
    hueco = np.max(np.diff(idx)) * 256 / SR
    if hueco > 0.9: return f"silencio interno {hueco:.1f}s"
    return ""

def toma(texto, q):
    w = m.generate(texto, language_id="es", audio_prompt_path=REFQ if q else REFA,
                   exaggeration=0.7 if q else 0.6, cfg_weight=0.5 if q else 0.7,
                   temperature=0.6 if q else 0.5).squeeze().cpu().numpy()
    return w

def entiende(w, ref):
    w16 = librosa.resample(w, orig_sr=SR, target_sr=16000)
    segs, _ = asr.transcribe(w16, language="es", beam_size=5)
    hyp = " ".join(s.text for s in segs).strip()
    return SequenceMatcher(None, norm(ref), norm(hyp)).ratio(), hyp

def limpia(w):
    w = sosfiltfilt(HP, w).astype(np.float32)
    w, _ = librosa.effects.trim(w, top_db=38, frame_length=1024, hop_length=128)
    f = int(0.015 * SR); ramp = np.linspace(0, 1, f, dtype=np.float32)
    w[:f] *= ramp; w[-f:] *= ramp[::-1]
    rms = np.sqrt(np.mean(w[np.abs(w) > 0.01] ** 2)) if np.any(np.abs(w) > 0.01) else 1
    w = w * (0.08 / rms)
    pk = np.max(np.abs(w));
    if pk > 0.89: w = w * (0.89 / pk)
    return np.concatenate([np.zeros(int(0.15 * SR), np.float32), w, np.zeros(int(0.35 * SR), np.float32)])

informe = []
partes = []
for i, t in enumerate(T, 1):
    q = t.strip().endswith("?")
    mejor = None; intentos = 0
    for texto in variantes(t):
        for _ in range(3 if texto == t else 2):
            intentos += 1
            w = toma(texto, q)
            d = defectos(w)
            sim, hyp = entiende(w, t)
            dur = len(w) / SR; cps = len(t) / max(dur, 0.1)
            if not (6 < cps < 22): d = d or f"ritmo raro {cps:.0f} letras/s"
            score = sim - (0.2 if d else 0)
            if mejor is None or score > mejor[0]: mejor = (score, sim, w, hyp, d, texto)
            if sim >= 0.98 and not d: break
        if mejor[1] >= 0.98 and not mejor[4]: break
    _, sim, w, hyp, d, texto = mejor
    w = limpia(w)
    sf.write(f"{OUT}/{i:02d}.wav", w, SR)
    partes.append(w)
    linea = f"{i:02d} {'OK ' if sim >= 0.95 and not d else 'REVISAR'} {sim:.2f} tomas={intentos} | {t} | oído: {hyp}" + (f" | {d}" if d else "") + (f" | escrito como: {texto}" if texto != t else "")
    print(linea, flush=True); informe.append(linea)

voz = np.concatenate(partes)
sf.write(f"{OUT}/voz.wav", voz, SR)
open(f"{OUT}/informe.txt", "w").write("\n".join(informe) + f"\nDuración total: {len(voz)/SR:.1f} s\n")
print(f"FIN · {len(voz)/SR:.1f} s · {OUT}/voz.wav", flush=True)
