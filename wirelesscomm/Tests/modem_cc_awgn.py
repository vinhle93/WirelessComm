"""

Test a simple chain of digital communication

 Source
    |
    |
Convolutional Encoding
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
Viterbi Decoding
    |
    |
Sink => BER, FER

"""

from numpy import arange, zeros, sqrt, sin, pi
#from numpy.random import seed
import matplotlib.pyplot as plt
from scipy.special import erfc

from wirelesscomm.source import Source
from wirelesscomm.sink import Sink
from wirelesscomm.modems.modems import QPSKModem, PSK8Modem
from wirelesscomm.channels.channels import AWGNChannel
from wirelesscomm.channelcodes.convolutionalcodes import ConvolutionalCode


frame_size = 600

source = Source(frame_size)
sink = Sink(source)
convcode = ConvolutionalCode(4,[13,15])
modem = QPSKModem()
awgn_chan = AWGNChannel()

EbN0dB = arange(0.0,3.1,0.5)
BER = zeros(len(EbN0dB))
FER = zeros(len(EbN0dB))


for i in range(len(EbN0dB)):
    awgn_chan.set_EbN0_dB(EbN0dB[i], channel_code_rate = convcode.code_rate,\
                         bits_per_modulated_symbol = modem.bits_per_modulated_symbol)
    while (sink.number_of_frame_errors < 100):
        data = source.generate()
        encoded_data = convcode.encode(data)
        modulated_symbols = modem.modulate(encoded_data)
        rcv_symbols = awgn_chan.propagate(modulated_symbols)
        demodulated_symbols = modem.demodulate(rcv_symbols, awgn_chan.noise_var, 'soft')
        decoded_data = convcode.viterbi_decode(demodulated_symbols)
        sink.receive_decoded_frame(decoded_data)
    BER[i], FER[i] = sink.get_error_rate()
    print(EbN0dB[i],sink.number_of_frames,sink.number_of_bit_errors,sink.number_of_frame_errors,BER[i],FER[i])
    sink.reset_error_meter()


EbN0linear = 10**(EbN0dB/10.0)
theoretical_uncoded_BER = None
if modem.bits_per_modulated_symbol < 3:
    theoretical_uncoded_BER = 0.5*erfc(sqrt(EbN0linear))
else:
    m = modem.bits_per_modulated_symbol
    theoretical_uncoded_BER = 1/m * erfc(sqrt(EbN0linear*m) * sin(pi/(1 << m)))

fig = plt.figure(1)
fig.clear()
plt.semilogy(EbN0dB,BER,'-ro',label=str(convcode.code))
plt.semilogy(EbN0dB,theoretical_uncoded_BER,'-bo',label='Uncoded')
plt.title(modem.modem_name + " bit error rate")
plt.legend(loc='lower left')
plt.grid(True, which="both")
plt.show()
