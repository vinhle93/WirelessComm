"""

Test a chain of OFDM system

 Source
    |
    |
Modulation <--- QPSK, 8PSK, QAM
    |
    |
OFDM Modulation
    |
    |
Fading Channel <-- Rayleigh
    |
    |
AWGN channel <--- Tuning EbN0_dB
    |
    |
OFDM Demodulation
    |
    |
Demodulation <-- QPSK, 8PSK, QAM
    |
    |
Sink => BER, FER


========== Test description ===========

The simulation shows the Bit Error Rate performance of QPSK-OFDM system under fading channels.

The received signal of an OFDM system can be expressed as

    Y_i = H_i*X_i + W_i,    (1)

where Y_i is the ith output from the DFT block
X_i is the ith transmitted QPSK symbol,
H_i is the ith DFT component of the channel h,
and W_i is the ith DFT component of the noise vector.

Note that if w_i is i.i.d CN(0,N0), then W_i is i.i.d CN(0,N0)

Since h is a random vector, H_i is a random variable with
    E[|H_i|^2] = E[||h||^2] = P_h


The error probability of detecting X_i (QPSK) given the system in (1) is:
    pe(detect X_i) = Q(sqrt(2 * |H_i|^2 * EbN0)),
where EbN0 is the average received signal-to-noise ratio per symbol time.

For fading channel, with Rayleigh taps, we have 
    H_i ~ CN(0,P_h).
The overall error probability is
    pe = E_H[Q(sqrt(2 * |H| * EbN0))] = 0.5 * (1 - sqrt(P_h*EbN0 / (1 + P_h*EbN0))).
At high EbN0,
    sqrt(P_h*EbN0 / (1 + P_h*EbN0)) -> 1 - 1/(2*P_h*EbN0)
Thus,
    pe(high EbN0) = 1/(4*P_h*EbN0).
======================================
"""

from numpy import arange, zeros, sqrt, sin, pi, array, conjugate
from numpy.fft import fft
from numpy.random import seed
import matplotlib.pyplot as plt

from wirelesscomm.source import Source
from wirelesscomm.sink import Sink
from wirelesscomm.modems.modems import QPSKModem, PSK8Modem
from wirelesscomm.modems.ofdm import OfdmModem_80211a, OfdmModem
from wirelesscomm.channels.channels import AWGNChannel, SISOFadingChannel, SISOIndoor80211aChannel


def Energy(x):
    return sum(conjugate(x)*x)

frame_size = 52*2

source = Source(frame_size)

sink = Sink(source)

modem = QPSKModem()

ofdmModem = OfdmModem_80211a()

fading_chan = SISOIndoor80211aChannel()
Gain_h = sum(fading_chan.pdp)

awgn_chan = AWGNChannel()


EbN0dB = arange(0.0,30.1,5.0)
BER = zeros(len(EbN0dB))
asymtotic_BER = zeros(len(EbN0dB))

for i in range(len(EbN0dB)):
    EbN0 = 10**(EbN0dB[i]/10.0)
    
    # SNR = E_rcv/N0 = Gain_h * EbN0 * r * M * (Nused/Nfft) * Nfft
    # The last multiplication of Nfft is because IFFT(X) = 1/Nfft * sum(..)
    # has a (1/Nfft) multiplication.
    # If we assume the average received power of symbol = 1,
    # N0 = 1 / (Gain_h * EbN0 * r * M * Nused)
    noise_power = 1 / (Gain_h * EbN0 * 1.0 * modem.bits_per_modulated_symbol * len(ofdmModem.data_subcarriers))
    awgn_chan.set_noise_power(noise_power)
    
    while (sink.number_of_frame_errors < 100):
        data = source.generate()
        
        modulated_sym = modem.modulate(data)
        ofdm_sym = ofdmModem.modulate(modulated_sym)
        
        faded_sym = fading_chan.propagate(ofdm_sym)
        rcv_sym = awgn_chan.propagate(faded_sym)
        
        deofdm_sym = ofdmModem.demodulate(rcv_sym)
        H = fft(fading_chan.h, ofdmModem.fft_size)
        equalized_sym = deofdm_sym / H[ofdmModem.data_subcarriers]
        
        decoded_data = modem.demodulate(equalized_sym,demod_type='hard')
        sink.receive_decoded_frame(decoded_data)
        
    BER[i], FER = sink.get_error_rate()
    asymtotic_BER[i] = 1/(4 * Gain_h * EbN0)
    print(EbN0dB[i],BER[i],asymtotic_BER[i])
    sink.reset_error_meter()


fig = plt.figure(1)
fig.clear()
plt.semilogy(EbN0dB,BER,'-ro',label='Monte-Carlo')
plt.semilogy(EbN0dB,asymtotic_BER,'-bo',label='Asymtotic')
plt.title("QPSK-OFDM bit error rate in fading channel")
plt.legend(loc='lower left')
plt.grid(True, which="both")
plt.show()