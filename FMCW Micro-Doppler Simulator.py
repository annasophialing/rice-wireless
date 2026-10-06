import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib", "scipy"])

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, stft

plt.rcParams.update({"figure.figsize": (8, 3.6), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler(color=["lightpink", "cyan", "blue", "lemonchiffon"])

# base parameters from table
c  = 3e8          # speed of light, m/s
f0 = 77e9         # carrier frequency, Hz
B  = 3.072e9      # chirp bandwidth, Hz
T  = 25.6e-6      # chirp duration, s
fs = 10e6         # ADC sampling rate, Hz
N  = 256          # samples per chirp

# derived quantities
S     = B / T  # chirp slope in Hz/s
lam   = c / f0  # wavelength in m
dR    = c / (2 * B)  # range resolution in m
R_max = c * fs / (2 * S)  # max range for a complex ADC (highest beat freq = fs)
t     = np.arange(N) / fs  # the N fast-time sample instants: 0, 1/fs, 2/fs, ...

print(f"S      = {S:.3e} Hz/s")
print(f"lambda = {lam*1e3:.3f} mm")
print(f"dR     = {dR*100:.3f} cm")
print(f"R_max  = {R_max:.2f} m")
print(f"N/fs   = {N/fs*1e6:.1f} us  (should equal T = {T*1e6:.1f} us)")

print(" R(m)   tau(ns)   f_b(kHz)   k_exact   k_peak   R_FFT(m)   err(cm)   RVP(deg)")
for R in [1, 2, 3, 4]:
    tau    = 2 * R / c  # round-trip delay
    fb     = S * tau  # beat frequency
    k      = fb / (fs / N)  # fractional bin index
    k_pk   = round(k)  # the bin the FFT peak will land on
    R_fft  = k_pk * dR  # range reported by that bin
    rvp    = 0.5 * S * tau**2 * 360  # residual video phase in degrees
    print(f"{R:4.0f} {tau*1e9:9.2f} {fb/1e3:10.0f} {k:9.2f} {k_pk:8d} {R_fft:10.4f} {(R_fft-R)*100:9.2f} {rvp:9.2f}")

# Q4: why 1 mm is invisible in range but obvious in phase
d = 1e-3
print(f"\n1 mm move = {d/dR:.3f} of a range bin, but {4*np.pi*d/lam:.2f} rad "
      f"({np.degrees(4*np.pi*d/lam):.0f} deg) of phase")

def if_signal(R, amp=1.0):
    '''Complex IF samples (length N) for one point reflector at range R (m).'''
    tau   = 2 * R / c  # round-trip delay
    phase = 2 * np.pi * (f0 * tau + S * tau * t - 0.5 * S * tau**2)  # equation (1), in radians
    return amp * np.exp(1j * phase)

x = if_signal(2.0)
print(x.shape, x.dtype)   # expect (256,) complex128

range_axis = np.arange(N) * (fs / N) * c / (2 * S)  # bin index -> frequency -> meters
print(f"spacing = {range_axis[1]*100:.3f} cm,  last bin = {range_axis[-1]:.2f} m")

X = np.fft.fft(if_signal(2.0))
plt.plot(range_axis, np.abs(X), ".-")
plt.axvline(2.0, ls="--", color="C1", label="true R")
plt.xlim(0, 5); plt.xlabel("range (m)"); plt.ylabel("|FFT|"); plt.title("Range FFT, one target at 2 m"); plt.legend()
plt.show()

print(" R_theory   R_FFT    error(cm)  within dR?")
for R in [1, 2, 3, 4]:
    X     = np.fft.fft(if_signal(R))
    R_fft = range_axis[np.argmax(np.abs(X))]  # range of the tallest bin
    err   = R_fft - R
    print(f"{R:8.2f} {R_fft:8.4f} {err*100:10.2f}   {abs(err) < dR}")
    assert abs(err) < dR, "range error exceeds one bin, check if_signal / range_axis"

for R in [1, 2, 3, 4]:
    plt.plot(range_axis, np.abs(np.fft.fft(if_signal(R))), label=f"R = {R} m")
plt.xlim(0, 5); plt.xlabel("range (m)"); plt.ylabel("|FFT|"); plt.title("Static targets at 1–4 m"); plt.legend()
plt.show()

targets = [(1.5, 1.0), (2.3, 0.6), (3.1, 0.3)]    # (range in m, amplitude a_k)

x3 = sum(if_signal(R, a) for R, a in targets)  # add up one if_signal per target

mag_rect = np.abs(np.fft.fft(x3))
plt.plot(range_axis, mag_rect, ".-")
for R, a in targets: plt.axvline(R, ls="--", color="C1", alpha=.6)
plt.xlim(0, 5); plt.xlabel("range (m)"); plt.ylabel("|FFT|"); plt.title("Three reflectors (linear scale)")
plt.show()

win      = np.hanning(N)
mag_hann = np.abs(np.fft.fft(x3 * win))  # multiply by the window BEFORE the FFT

db = lambda m: 20 * np.log10(m / m.max() + 1e-12)
plt.plot(range_axis, db(mag_rect), label="rectangular (no window)")
plt.plot(range_axis, db(mag_hann), label="Hann window")
for R, a in targets: plt.axvline(R, ls="--", color="gray", alpha=.5)
plt.xlim(0, 5); plt.ylim(-60, 3); plt.xlabel("range (m)"); plt.ylabel("dB (rel. max)")
plt.title("Same signal in dB: sidelobes vs window"); plt.legend()
plt.show()

peaks, _ = find_peaks(db(mag_hann), height=-20)  # keep peaks within 20 dB of the max
print("detected ranges (m):", np.round(range_axis[peaks], 3))
print("true ranges     (m):", [R for R, a in targets])
assert len(peaks) == len(targets), "wrong number of peaks"

range_fine = np.arange(8*N) * (fs/(8*N)) * c / (2*S)     # range axis for an 8x zero-padded FFT
for d in [0.03, 0.06, 0.12]:
    x2 = if_signal(2.0) + if_signal(2.0 + d)
    plt.plot(range_fine, np.abs(np.fft.fft(x2, n=8*N)), label=f"d = {d*100:.0f} cm")
plt.xlim(1.8, 2.3); plt.xlabel("range (m)"); plt.ylabel("|FFT|"); plt.title("Two equal reflectors d apart"); plt.legend()
plt.show()

# slow-time setup
Tc     = 1e-3                       # chirp repetition interval, s  (PRF = 1 kHz)
M      = 4000                       # number of chirps  -> 4 s total
t_slow = np.arange(M) * Tc  # slow-time instants of each chirp

# target motion (simple harmonic)
R0, A, fm = 2.0, 0.05, 1.0          # 2 m away, +/-5 cm, 1 Hz
R_t = R0 + A * np.sin(2 * np.pi * fm * t_slow)  # equation (3)

v_peak  = 2 * np.pi * fm * A  # max of dR/dt
fD_peak = 2 * v_peak / lam  # Doppler at peak velocity
v_max   = lam / (4 * Tc)  # max unambiguous velocity
print(f"v_peak = {v_peak:.3f} m/s,  f_D peak = {fD_peak:.0f} Hz")
print(f"v_max  = {v_max:.3f} m/s,  Doppler limit = +/-{1/(2*Tc):.0f} Hz")
assert v_peak < v_max, "motion too fast for this Tc -> Doppler will alias"

plt.plot(t_slow, R_t * 100)
plt.xlabel("time (s)"); plt.ylabel("range (cm)"); plt.title("1. True range trajectory R(t)")
plt.show()

X = np.array([if_signal(R) for R in R_t])  # one row per chirp, using stop-and-hop
print(X.shape)   # expect (4000, 256)

RP = np.fft.fft(X * np.hanning(N)[None, :], axis=1)  # range FFT on each chirp (row)

rtm_db = 20 * np.log10(np.abs(RP) / np.abs(RP).max() + 1e-12)
plt.pcolormesh(t_slow[::10], range_axis, rtm_db[::10].T, vmin=-40, vmax=0, shading="nearest", cmap="viridis")
plt.plot(t_slow, R_t, "w--", lw=1, label="true R(t)")
plt.colorbar(label="dB"); plt.ylim(R0 - 0.3, R0 + 0.3)
plt.xlabel("time (s)"); plt.ylabel("range (m)"); plt.title("2. Range–Time Map"); plt.legend(loc="upper right")
plt.show()

k0   = np.argmin(np.abs(range_axis - R0))  # index of the range bin closest to R0
slow = RP[:, k0]  # that bin's complex value, one per chirp

disp = np.unwrap(np.angle(slow)) * lam / (4 * np.pi)  # phase -> meters
disp -= disp.mean()
plt.plot(t_slow, (R_t - R0) * 100, lw=3, alpha=.4, label="true R(t) − R0")
plt.plot(t_slow, disp * 100, "--", label="recovered from phase")
plt.xlabel("time (s)"); plt.ylabel("displacement (cm)"); plt.title(f"Phase of range bin {k0} ({range_axis[k0]:.3f} m)"); plt.legend()
plt.show()

f_d, t_stft, Z = stft(slow, fs=1/Tc, window="hann", nperseg=128, noverlap=120, return_onesided=False)  # STFT of `slow`: sample rate 1/Tc, Hann, 128-chirp window, 120 overlap, two-sided
f_d = np.fft.fftshift(f_d)
Z   = np.fft.fftshift(Z, axes=0)

fD_theory = (2 / lam) * 2 * np.pi * fm * A * np.cos(2 * np.pi * fm * t_slow)  # (2/lambda)*dR/dt

P = 20 * np.log10(np.abs(Z) / np.abs(Z).max() + 1e-12)
plt.pcolormesh(t_stft, f_d, P, vmin=-40, vmax=0, shading="nearest", cmap="viridis")
plt.plot(t_slow, fD_theory, "w--", lw=1, label="theory (2/λ)·dR/dt")
plt.colorbar(label="dB"); plt.ylim(-300, 300)
plt.xlabel("time (s)"); plt.ylabel("Doppler (Hz)"); plt.title("3. Micro-Doppler spectrogram"); plt.legend(loc="upper right")
plt.show()