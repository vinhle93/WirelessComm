"""

Test a simple chain of digital communication

 Source
    |
    |
Modulation <--- QPSK, 8PSK, QAM
    |
    |
AWGN channel <--- Tuning EbN0_dB
    |
    |
Demodulation <-- QPSK, 8PSK, QAM
    |
    |
  Sink => BER, FER

"""

from numpy import arange, zeros, sqrt
import matplotlib.pyplot as plt
from scipy.special import erfc

from wirelesscomm.source import Source
from wirelesscomm.sink import Sink
from wirelesscomm.modems.modems import QPSKModem
from wirelesscomm.channels.channels import AWGNChannel

frame_size = 500

source = Source(frame_size)
sink = Sink(source)
modem = QPSKModem()
awgn_chan = AWGNChannel()

EbN0dB = arange(0.0,6.1,1.0)
BER = zeros(len(EbN0dB))
FER = zeros(len(EbN0dB))

for i in range(len(EbN0dB)):
    awgn_chan.set_EbN0_dB(EbN0dB[i], bits_per_modulated_symbol=modem.bits_per_modulated_symbol)
    while (sink.number_of_frame_errors < 50):
        data = source.generate()
        modulated_data = modem.modulate(data)
        rcv_data = awgn_chan.propagate(modulated_data)
        decoded_data = modem.demodulate(rcv_data, awgn_chan.noise_var, 'hard')
        sink.receive_decoded_frame(decoded_data)
    BER[i], FER[i] = sink.get_error_rate()
    print(EbN0dB[i],sink.number_of_frames,sink.number_of_bit_errors,sink.number_of_frame_errors,BER[i],FER[i])
    sink.reset_error_meter()

EbN0linear = 10**(EbN0dB/10.0)
theoretical_qpsk_BER = 0.5*erfc(sqrt(EbN0linear))
print(theoretical_qpsk_BER)


fig = plt.figure(1)
fig.clear()
plt.semilogy(EbN0dB,BER,'-ro')
plt.semilogy(EbN0dB,theoretical_qpsk_BER,'-bo')
plt.grid(True, which="both")
plt.show()
