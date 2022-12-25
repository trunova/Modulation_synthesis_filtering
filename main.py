import scipy
import matplotlib.pyplot as plt
import math
import numpy as np

amplitude = 1
phi = 0
SAMPLE_RATE = 1000
periods = np.arange(0, 1, 1/SAMPLE_RATE)
fc = 4
fm = 1

def harmonic_signal(w, a=amplitude, phi=phi):
    return [a * np.sin(2 * np.pi * w * t + phi) for t in periods]


def meander(w):
    h_signal = harmonic_signal(w)
    return [amplitude if h_signal[i] > 0 else 0 for i in range(0, len(h_signal))]

def modulation_am():
    return harmonic_signal(25 * fc) * (amplitude + 0.5 * np.sign(np.sin(10 * periods * 2 * np.pi)))

def modulation_fm():
    m_sig = meander(4 * fm)
    h_sig = harmonic_signal(5 * fc)
    h_sig_2 = harmonic_signal(25 * fc)
    return [h_sig_2[i] if m_sig[i] == 0 else h_sig[i] for i in range(len(m_sig))]


def modulation_phm():
    m_sig = np.sign(harmonic_signal(15))
    h_sig = harmonic_signal(30)
    h_sig_2 = harmonic_signal(30, a=-1)
    return [h_sig[i] if m_sig[i] > 0 else h_sig_2[i] for i in range(len(m_sig))]


def spectrum(signal):
    spectrum_y = scipy.fft.fft(signal)
    spectrum_x = scipy.fft.fftfreq(SAMPLE_RATE, 1 / SAMPLE_RATE)
    return [spectrum_x[range(1, 300)], spectrum_y[range(1, 300)]]


def synthesizing():
    modulation = modulation_am()
    y_f = np.abs(scipy.fft.rfft(modulation))
    x_f = scipy.fft.rfftfreq(len(periods), 1 / len(periods))
    syntez = np.fft.ifft(np.array([0 if np.abs(y) < 100 else np.abs(y) for y in y_f]))
    return np.real(syntez), x_f / (len(periods) / 2)


def filter_signal(cut_signal):
    filtering = []
    values = scipy.signal.butter(25, 0.01)
    for i in scipy.signal.filtfilt(values[0], values[1], cut_signal):
        filtering.append((np.sign(i) + 1) / 2)
    return filtering


if __name__ == "__main__":
    am_signal = modulation_am()
    fm_signal = modulation_fm()
    phm_signal = modulation_phm()

    spectrum_am = spectrum(am_signal)
    spectrum_fm = spectrum(fm_signal)
    spectrum_phm = spectrum(phm_signal)

    meander = meander(10 * fm)

    synth, time_synyh = synthesizing()
    filt = filter_signal(synth)

    colors = ['#3caa3c', '#a330bf', '#cc2f98']

    plt.figure(3)
    plt.figtext(0.3, 0.9, 'Синтезированный сигнал и отфильтрованый спектр')
    plt.subplot(2, 1, 1)
    plt.plot(time_synyh, synth)
    plt.subplot(2, 1, 2)
    plt.plot(time_synyh, filt)
    plt.grid(True)

    plt.figure(2)
    plt.figtext(0.3, 0.9, 'Спектры модуляций')
    i = 0
    for spectrum in [spectrum_am, spectrum_fm, spectrum_phm]:
        plt.subplot(3, 1, i + 1)
        plt.plot(spectrum[0], abs(spectrum[1]), colors[i])
        i += 1
    plt.grid(True)

    plt.figure(1)
    plt.figtext(0.3, 0.9, 'Амплитудная, частотная и фазовая модуляции')
    i = 0
    for signal in [am_signal, fm_signal, phm_signal]:
        plt.subplot(3, 1, i+1)
        plt.plot(periods, signal, colors[i])
        i += 1
    plt.grid(True)

    plt.show()