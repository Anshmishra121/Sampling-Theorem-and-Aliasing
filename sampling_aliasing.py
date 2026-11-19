"""
Signals & Systems Innovative Assessment
Topic: Sampling Theorem and Aliasing
Student: Rohitanshu Kharbanda
Roll No.: 202410101150241

Run:
    pip install -r requirements.txt
    python sampling_aliasing.py

The script generates four PNG figures in the "figures" directory.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- Signal definition --------------------
A = 1.0                 # amplitude
f0 = 5.0                # original sinusoid frequency in Hz
duration = 1.0          # observation duration in seconds
fs_high = 1000.0        # high-rate grid used to display continuous-time signal
t = np.arange(0, duration, 1 / fs_high)
x = A * np.sin(2 * np.pi * f0 * t)

# Sampling frequencies: below, at, and above the Nyquist rate
sampling_rates = [6.0, 10.0, 40.0]
case_names = [
    "Undersampling (fs = 6 Hz)",
    "Nyquist-rate sampling (fs = 10 Hz)",
    "Oversampling (fs = 40 Hz)",
]

def sinc_reconstruct(sample_times, samples, query_times, fs):
    """Ideal sinc interpolation for a uniformly sampled, band-limited signal."""
    return np.sum(
        samples[:, None] * np.sinc(fs * (query_times[None, :] - sample_times[:, None])),
        axis=0,
    )

# -------------------- Figure 1: sampling cases --------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
all_samples = []
for ax, fs, title in zip(axes, sampling_rates, case_names):
    ts = np.arange(0, duration, 1 / fs)
    xs = A * np.sin(2 * np.pi * f0 * ts)
    all_samples.append((ts, xs, fs))
    ax.plot(t, x, label="Original signal", linewidth=1.6)
    ax.stem(ts, xs, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Samples")
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Sampling a 5 Hz sinusoidal signal")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "sampling_cases.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# -------------------- Figure 2: aliasing --------------------
fs = 6.0
alias_frequency = abs(f0 - round(f0 / fs) * fs)
ts = np.arange(0, duration, 1 / fs)
xs = A * np.sin(2 * np.pi * f0 * ts)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, x, label=f"Original: {f0:g} Hz", linewidth=1.8)
ax.plot(
    t,
    A * np.sin(2 * np.pi * alias_frequency * t),
    "--",
    label=f"Alias-frequency illustration: {alias_frequency:g} Hz",
)
ax.stem(ts, xs, linefmt="C2-", markerfmt="C2o", basefmt=" ", label=f"Samples at {fs:g} Hz")
ax.set_title("Aliasing caused by undersampling")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "aliasing_demo.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# -------------------- Figure 3: sinc reconstruction --------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, (ts, xs, fs), title in zip(axes, all_samples, case_names):
    x_rec = sinc_reconstruct(ts, xs, t, fs)
    ax.plot(t, x, label="Original", linewidth=1.8)
    ax.plot(t, x_rec, "--", label="Sinc reconstruction", linewidth=1.4)
    ax.stem(ts, xs, linefmt="C2-", markerfmt="C2o", basefmt=" ", label="Samples")
    ax.set_title(title)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Ideal sinc reconstruction from finite sample records")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "reconstruction_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# -------------------- Figure 4: frequency spectrum --------------------
N = len(t)
frequencies = np.fft.rfftfreq(N, d=1 / fs_high)
spectrum = np.abs(np.fft.rfft(x)) / N * 2
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(frequencies, spectrum, linewidth=1.6)
ax.set_xlim(0, 30)
ax.set_title("Single-sided magnitude spectrum of the original signal")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "frequency_spectrum.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Simulation completed. Figures saved in:", OUTPUT_DIR.resolve())
print(f"Original signal frequency: {f0:.1f} Hz")
print(f"Nyquist rate: {2*f0:.1f} samples/second")
print("Sampling rates tested:", sampling_rates, "Hz")
