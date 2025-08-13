import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window

def save_spectrogram_pdf(x_iq, fs, n_fft=256, hop=64, win='hann', out='figures/spectrogram_example.pdf'):
    """
    x_iq: complex1d array (I + jQ)
    fs:   sample rate (Hz)
    """
    f, t, Z = stft(x_iq, fs=fs, nperseg=n_fft, noverlap=n_fft-hop, window=get_window(win, n_fft), return_onesided=True)
    P = 20*np.log10(np.maximum(np.abs(Z), 1e-12))  # dB
    plt.figure(figsize=(6,3))
    plt.imshow(P, aspect='auto', origin='lower', extent=[t[0], t[-1], f[0], f[-1]])
    plt.xlabel('Time (s)'); plt.ylabel('Frequency (Hz)')
    plt.title('STFT spectrogram')
    plt.colorbar(label='Magnitude (dB)')
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    plt.close()

# Example usage:
# x = your complex IQ array, fs = sample rate
# save_spectrogram_pdf(x, fs)
