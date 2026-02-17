import math
import random
import struct
import wave


def generate_glitch_lullaby(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    bpm = 78
    beat_dur = 60 / bpm
    total_samples = int(sample_rate * duration_sec)

    left = [0.0] * total_samples
    right = [0.0] * total_samples

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    def noise(amp=1.0):
        return (random.random() * 2 - 1) * amp

    for i in range(total_samples):
        t = i / sample_rate
        beat_time = t % beat_dur

        bass = 0.0
        if beat_time < 0.4:
            env = math.exp(-12 * beat_time)
            freq = 60 * math.exp(-4 * beat_time)
            bass = osc_sine(t, freq) * env * 0.6

        piano = 0.0
        bar_time = t % (beat_dur * 4)
        chord_root = 220
        if bar_time > beat_dur * 2:
            chord_root = 174.6
        if bar_time > beat_dur * 3:
            chord_root = 261.6

        note_dur = beat_dur / 4
        note_idx = int(t / note_dur)
        note_t = t % note_dur

        if note_idx % 2 == 0:
            freq = chord_root * (1.5 if note_idx % 3 == 0 else 1.0)
            if note_idx % 7 == 0:
                freq *= 1.2

            env = math.exp(-8 * note_t)
            mod = osc_sine(t, freq * 2.0) * env * 2.0
            piano = osc_sine(t, freq + mod) * env * 0.25

        warble = math.sin(t * 0.5) * 0.05

        is_glitch = (t % 8.0) > 7.5
        glitch_mod = 1.0
        if is_glitch:
            chop_rate = 20.0
            if (t * chop_rate) % 1.0 > 0.5:
                glitch_mod = 0.0
            piano += noise(0.1)

        mix = (bass + piano) * glitch_mod

        pan = math.sin(t * 0.3) * 0.3
        l_out = mix * (0.7 - pan)
        r_out = mix * (0.7 + pan)

        hiss = noise(0.005)

        left[i] = max(-1.0, min(1.0, l_out + hiss))
        right[i] = max(-1.0, min(1.0, r_out + hiss))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        packed = bytearray()
        for i in range(total_samples):
            l_int = int(left[i] * 32767)
            r_int = int(right[i] * 32767)
            packed += struct.pack("<hh", l_int, r_int)

        wf.writeframes(packed)


def generate_dialectic_tones(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    packed_data = bytearray()

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    for i in range(total_samples):
        t = i / sample_rate

        # Dialectic competition between two frequencies
        # US: 440Hz (A4) vs JA: 432Hz (Ancient/Verdi)
        freq_us = 440
        freq_ja = 432

        # Shifting dominance
        mix_bal = 0.5 + math.sin(t * 0.2) * 0.5

        # Glitch pulses
        glitch = 1.0 if (t * 8) % 1.0 > 0.1 else 0.0

        sig_us = osc_sine(t, freq_us) * (1 - mix_bal)
        sig_ja = osc_sine(t, freq_ja) * mix_bal

        mix = (sig_us + sig_ja) * 0.5 * glitch

        # Distortion
        mix = max(-0.8, min(0.8, mix * 1.5))

        packed_data += struct.pack("<hh", int(mix * 32767), int(mix * 32767))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(packed_data)


def generate_survival_receipt(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    packed_data = bytearray()

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    for i in range(total_samples):
        t = i / sample_rate

        # Heavy sub-bass heartbeat (survival focus)
        bass = osc_sine(t, 55 + math.sin(t * 0.1) * 5) * 0.6 * math.exp(-5 * (t % 0.76))

        # Minimal piano (Mage focus)
        piano = 0.0
        if (t * 2) % 1.0 > 0.8:
            piano = (
                osc_sine(t, 440 * (1.5 if t % 2 > 1 else 1))
                * 0.2
                * math.exp(-10 * (t % 0.5))
            )

        # Glitch (Calamity echoes)
        glitch = 0.0
        if (t * 12) % 1.0 < 0.05:
            glitch = (random.random() * 2 - 1) * 0.1

        mix = bass + piano + glitch
        mix = max(-0.9, min(0.8, mix))

        packed_data += struct.pack("<hh", int(mix * 32767), int(mix * 32767))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(packed_data)


def generate_core_ignition(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    packed_data = bytearray()

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    def noise(amp=1.0):
        return (random.random() * 2 - 1) * amp

    for i in range(total_samples):
        t = i / sample_rate

        # Shifting pulse (The Decision Field)
        pulse = osc_sine(t, 92 / 60)  # 92 BPM pulse
        pulse_amp = 0.5 + pulse * 0.2

        # High-fidelity VL harmonics
        # Ascending series of sine waves representing "Sight"
        harmonic1 = osc_sine(t, 880 + math.sin(t * 0.1) * 20) * 0.1
        harmonic2 = osc_sine(t, 1320 + math.cos(t * 0.15) * 30) * 0.05
        harmonic3 = osc_sine(t, 1760 + math.sin(t * 0.2) * 40) * 0.02

        # Ignition roar (filtered noise)
        roar = 0.0
        if (t % 4.0) < 0.5:  # Every 4 seconds
            env = math.exp(-10 * (t % 4.0))
            roar = noise(0.3) * env

        mix = (harmonic1 + harmonic2 + harmonic3 + roar) * pulse_amp
        mix = max(-0.9, min(0.8, mix))

        packed_data += struct.pack("<hh", int(mix * 32767), int(mix * 32767))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(packed_data)


def generate_weaver_psalm(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    packed_data = bytearray()

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    def noise(amp=1.0):
        return (random.random() * 2 - 1) * amp

    for i in range(total_samples):
        t = i / sample_rate

        # Shimmering high-end (Weaver focus)
        # Fast modulation representing "weaving"
        mod = osc_sine(t, 10) * 50
        weaver = osc_sine(t, 1000 + mod) * 0.15

        # Crystal harmonics
        crystal = (
            (osc_sine(t, 2000) + osc_sine(t, 3000))
            * 0.05
            * (0.5 + 0.5 * math.sin(t * 2))
        )

        # Glitch-sweeps
        sweep = 0.0
        if (t % 2.0) < 0.2:
            freq = 5000 * math.exp(-20 * (t % 2.0))
            sweep = osc_sine(t, freq) * 0.1

        # Deep resonance (The Core beat)
        beat = osc_sine(t, 55) * 0.4 * math.exp(-8 * (t % (60 / 102)))

        mix = weaver + crystal + sweep + beat
        mix = max(-0.9, min(0.8, mix))

        packed_data += struct.pack("<hh", int(mix * 32767), int(mix * 32767))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(packed_data)


def generate_trembling_hinge(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    packed_data = bytearray()

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    def noise(amp=1.0):
        return (random.random() * 2 - 1) * amp

    for i in range(total_samples):
        t = i / sample_rate

        # Fractured beats (105 BPM)
        beat_dur = 60 / 105
        beat_time = t % beat_dur
        kick = (
            osc_sine(t, 60 * math.exp(-10 * beat_time)) * 0.5 * math.exp(-5 * beat_time)
        )

        # Snare/Fracture (every 2nd beat)
        snare = 0.0
        if (t % (beat_dur * 2)) > beat_dur:
            if beat_time < 0.1:
                snare = noise(0.2) * math.exp(-20 * beat_time)

        # Reversed choir effect (Swell)
        choir = 0.0
        swell_dur = 2.0
        swell_t = (t % swell_dur) / swell_dur
        # Swell amplitude (reversed profile)
        choir_amp = math.pow(swell_t, 4) * 0.2
        choir = (osc_sine(t, 440) + osc_sine(t, 554) + osc_sine(t, 659)) * choir_amp

        # Radio static (intermittent)
        static = noise(0.02) if math.sin(t * 0.5) > 0.8 else 0.0

        mix = kick + snare + choir + static
        mix = max(-0.9, min(0.8, mix))

        packed_data += struct.pack("<hh", int(mix * 32767), int(mix * 32767))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(packed_data)


def generate_witness_collapse(filename: str, duration_sec: float = 30.0):
    sample_rate = 44100
    total_samples = int(sample_rate * duration_sec)
    packed_data = bytearray()

    def osc_sine(t, freq):
        return math.sin(2 * math.pi * freq * t)

    def noise(amp=1.0):
        return (random.random() * 2 - 1) * amp

    for i in range(total_samples):
        t = i / sample_rate

        # 78 BPM Pulse
        beat_dur = 60 / 78
        beat_time = t % beat_dur

        # Soft sub-bass heartbeat
        bass = osc_sine(t, 50 + math.sin(t * 0.05) * 2) * 0.5 * math.exp(-4 * beat_time)

        # Minimal piano arpeggio
        piano = 0.0
        if (t * 4) % 1.0 < 0.1:
            p_freq = 220 * (1.5 if (t // 2) % 2 == 0 else 1.25)
            piano = osc_sine(t, p_freq) * 0.2 * math.exp(-8 * (t % 0.25))

        # Tape warble (LFO on pitch)
        warble = 1.0 + math.sin(t * 2.0) * 0.005

        # Stutters on "anchor" and "tax" (Periodic glitch)
        glitch = 1.0
        if (t % 4.0) > 3.8:  # "Tax" stutter
            if (t * 20) % 1.0 > 0.5:
                glitch = 0.0
        elif (t % 4.0) > 1.8 and (t % 4.0) < 2.0:  # "Anchor" stutter
            if (t * 30) % 1.0 > 0.5:
                glitch = 0.0

        # Whispered lead (Noise modulation)
        whisper = noise(0.02) * (0.5 + 0.5 * math.sin(t * 0.5))

        mix = (bass + piano + whisper) * glitch * warble
        mix = max(-0.9, min(0.8, mix))

        packed_data += struct.pack("<hh", int(mix * 32767), int(mix * 32767))

    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(packed_data)


if __name__ == "__main__":
    import sys

    out = "eta_mu_glitch_lullaby.part_66.wav"
    if len(sys.argv) > 1:
        out = sys.argv[1]

    if "dialectic" in out:
        print(f"Generating dialectic tones {out}...")
        generate_dialectic_tones(out)
    elif "survival" in out:
        print(f"Generating survival receipt {out}...")
        generate_survival_receipt(out)
    elif "core_ignition" in out:
        print(f"Generating core ignition {out}...")
        generate_core_ignition(out)
    elif "weaver" in out:
        print(f"Generating weaver psalm {out}...")
        generate_weaver_psalm(out)
    elif "hinge" in out:
        print(f"Generating trembling hinge {out}...")
        generate_trembling_hinge(out)
    elif "witness_collapse" in out:
        print(f"Generating witness collapse {out}...")
        generate_witness_collapse(out)
    else:
        print(f"Generating lullaby {out}...")
        generate_glitch_lullaby(out)
    print("Done.")
