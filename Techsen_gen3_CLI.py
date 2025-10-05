import numpy as np
import sounddevice as sd
import aubio
import time
from collections import deque
import argparse
# -------------------------------
# CONFIG
# -------------------------------
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
HOP_SIZE = 1024

# Swara → semitone map with komal, tivra, octaves
def swara_to_semitone(swara):
    mapping = {
        # Middle octave
        "Sa": 0,
        "komal Re": 1,
        "Re": 2,
        "komal Ga": 3,
        "Ga": 4,
        "Ma": 5,
        "Tivra Ma": 6,
        "Pa": 7,
        "komal Dha": 8,
        "Dha": 9,
        "komal Ni": 10,
        "Ni": 11,
        "Sa'": 12,
        # Lower octave
        "Sa↓": -12, "komal Re↓": -11, "Re↓": -10, "komal Ga↓": -9, "Ga↓": -8,
        "Ma↓": -7, "Tivra Ma↓": -6, "Pa↓": -5, "komal Dha↓": -4, "Dha↓": -3,
        "komal Ni↓": -2, "Ni↓": -1,
        # Upper octave
        "Sa↑": 12, "komal Re↑": 13, "Re↑": 14, "komal Ga↑": 15, "Ga↑": 16,
        "Ma↑": 17, "Tivra Ma↑": 18, "Pa↑": 19, "komal Dha↑": 20, "Dha↑": 21,
        "komal Ni↑": 22, "Ni↑": 23, "Sa'↑": 24
    }
    return mapping.get(swara, 0)

# Base raagas
RAAGA = {
    "bhairav":  [swara_to_semitone(s) for s in ["Sa", "komal Re", "Ga", "Ma", "Pa", "komal Dha", "Ni", "Sa'"]],
    "bhairavi": [swara_to_semitone(s) for s in ["Sa", "komal Re", "komal Ga", "Ma", "Pa", "komal Dha", "komal Ni", "Sa'"]],
    "bhupali":  [swara_to_semitone(s) for s in ["Sa", "Re", "Ga", "Pa", "Dha", "Sa'"]],
    "malkauns": [swara_to_semitone(s) for s in ["Sa", "komal Ga", "Ma", "komal Dha", "komal Ni", "Sa'"]],
    "asavari":  [swara_to_semitone(s) for s in ["Sa", "Re", "komal Ga", "Ma", "Pa", "komal Dha", "komal Ni", "Sa'"]],
}

REFERENCE_SA = None
TANPURA_INTERVALS = [0, 7, 12]
SMOOTH_WINDOW = 6
HOLD_THRESHOLD = 4
BASE_MEEND_TIME = 0.08  # seconds per semitone

# -------------------------------
# ADD CUSTOM RAAGA FUNCTION
# -------------------------------
def add_raaga(name, swaras):
    RAAGA[name.lower()] = [swara_to_semitone(s) for s in swaras]
    print(f"[INFO] Added new raaga: {name}")

# -------------------------------
# Aubio pitch detector
# -------------------------------
pitch_detector = aubio.pitch("yin", BUFFER_SIZE, HOP_SIZE, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_silence(-40)

# -------------------------------
# Detect reference Sa
# -------------------------------
def detect_reference_sa(duration=3):
    print("[INFO] Listening to detect reference Sa... sing your Sa steadily now.")
    stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
    detected_pitches = []

    with stream:
        start = time.time()
        while time.time() - start < duration:
            audio, _ = stream.read(BUFFER_SIZE)
            samples = np.mean(audio, axis=1).astype(np.float32)
            pitch = pitch_detector(samples)[0]
            if pitch > 50:
                detected_pitches.append(pitch)

    if not detected_pitches:
        raise ValueError("Could not detect Sa, please sing louder/clearer.")

    median_pitch = np.median(detected_pitches)
    print(f"[INFO] Detected Sa ≈ {median_pitch:.2f} Hz")
    return median_pitch

# -------------------------------
# Tanpura
# -------------------------------
def tanpura_wave(freq, duration, amp=0.1):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    wave += 0.3 * np.sin(2 * np.pi * freq * 3 * t)
    return amp * wave

def play_tanpura(sa_freq):
    tanpura = np.zeros(int(SAMPLE_RATE * 2))
    for interval in TANPURA_INTERVALS:
        freq = sa_freq * (2 ** (interval / 12))
        tanpura += tanpura_wave(freq, 2, amp=0.02)
    return tanpura.astype(np.float32)

# -------------------------------
# Flute continuous oscillator
# -------------------------------
current_freq = 0.0
target_freq = 0.0
phase = 0.0
samples_remaining = 0

def flute_callback(outdata, frames, time_info, status):
    global phase, current_freq, target_freq, samples_remaining
    freqs = np.zeros(frames)

    if samples_remaining > 0:
        step = min(frames, samples_remaining)
        ramp = np.linspace(current_freq, target_freq, step, endpoint=False)
        freqs[:step] = ramp
        current_freq = ramp[-1]
        samples_remaining -= step
        if step < frames:
            freqs[step:] = current_freq
    else:
        freqs[:] = current_freq

    phase_increments = (2 * np.pi * freqs) / SAMPLE_RATE
    phases = np.cumsum(phase_increments) + phase
    phase = phases[-1] % (2 * np.pi)
    wave = np.sin(phases) + 0.5 * np.sin(2 * phases)
    wave *= 0.3
    outdata[:, 0] = wave.astype(np.float32)

# -------------------------------
# Raaga helper
# -------------------------------
def snap_to_raaga(semitone):
    valid = RAAGA[CURRENT_RAAGA]
    nearest = min(valid, key=lambda x: abs(x - semitone % 12))
    return 12 * (semitone // 12) + nearest

# -------------------------------
# Pitch tracking + micro-direction logic
# -------------------------------
last_note = None
pitch_buffer = deque(maxlen=SMOOTH_WINDOW)
stable_counter = 0

def input_callback(indata, frames, time_info, status):
    global last_note, target_freq, stable_counter, samples_remaining, current_freq
    mono = np.mean(indata, axis=1).astype(np.float32)
    pitch = pitch_detector(mono)[0]

    if pitch <= 0 or REFERENCE_SA is None:
        return

    semitones = 12 * np.log2(pitch / REFERENCE_SA)
    nearest = int(round(semitones))
    snapped = snap_to_raaga(nearest)

    pitch_buffer.append(snapped)
    median_note = int(np.median(pitch_buffer))

    # Direction detection (up or down)
    if len(pitch_buffer) >= 3:
        trend = np.sign(np.mean(np.diff(pitch_buffer)))
    else:
        trend = 0

    # Prevent octave jumps
    if last_note is not None and abs(median_note - last_note) > 12:
        return

    # Micro slide in direction of motion
    if trend != 0 and last_note is not None:
        median_note += trend * 0.2  # small fractional slide

    if last_note is None or abs(median_note - last_note) > 0.2:
        stable_counter += 1
        if stable_counter >= HOLD_THRESHOLD:
            freq = REFERENCE_SA * (2 ** (median_note / 12))
            interval = abs(median_note - (last_note if last_note else 0))
            glide_time = BASE_MEEND_TIME * max(1, interval)
            samples_remaining = max(int(glide_time * SAMPLE_RATE), 1)
            target_freq = freq
            print(f"Swara: {median_note:+.2f} → {freq:.2f} Hz (glide {glide_time:.2f}s, dir={trend:+})")
            last_note = median_note
            stable_counter = 0
    else:
        stable_counter = 0

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("▗▄▄▄▖▗▄▄▄▖ ▗▄▄▖▗▖ ▗▖ ▗▄▄▖▗▄▄▄▖▗▖  ▗▖")
    print("  █  ▐▌   ▐▌   ▐▌ ▐▌▐▌   ▐▌   ▐▛▚▖▐▌")
    print("  █  ▐▛▀▀▘▐▌   ▐▛▀▜▌ ▝▀▚▖▐▛▀▀▘▐▌ ▝▜▌")
    print("  █  ▐▙▄▄▖▝▚▄▄▖▐▌ ▐▌▗▄▄▞▘▐▙▄▄▖▐▌  ▐▌")
    REFERENCE_SA = detect_reference_sa()
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--raag",type = str)
    args = parser.parse_args()
    CURRENT_RAAGA=args.raag.lower()

    tanpura_buf = play_tanpura(REFERENCE_SA)
    sd.play(tanpura_buf, SAMPLE_RATE, loop=True)

    flute_stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE,
                                   blocksize=BUFFER_SIZE, callback=flute_callback)
    flute_stream.start()
    with sd.InputStream(callback=input_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=BUFFER_SIZE):
        print(f"[INFO] Live tracking started in Raaga {CURRENT_RAAGA}. Sing swaras!")
        
        while True:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping...")
                flute_stream.stop()
                sd.stop()
                break
