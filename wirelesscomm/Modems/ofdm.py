"""
OFDM Modulator-Demodulator Models
=================================

"""

from numpy import zeros, arange, array, concatenate, ndarray, log2, isin
from numpy.fft import fft, ifft

__all__ = ['OfdmModem','OfdmModem_80211a']

class OfdmModem():

    def __init__(self, fftsize):
        self._fft_size = fftsize
        self._data_subcarriers = arange(fftsize)
        self._zero_subcarriers = array([], dtype='int')
        self.cyclic_prefix_length = 0

    def modulate(self, input_symbols):
        if len(input_symbols) > len(self.data_subcarriers):
            raise ValueError("Length of input_symbols should be less or equal to the number of data subcarriers")

        # mapping input_symbols to data subcarriers, other subcarriers = 0
        input_ifft = zeros(self.fft_size,dtype='complex')
        for i in range(len(input_symbols)):
            input_ifft[self.data_subcarriers[i]] = input_symbols[i]

        # inverse fast fourier transform
        output_ifft = ifft(input_ifft,self.fft_size)

        # add cyclic prefix: ofdm = [x(N-Ng-1), ..., x(0), ..., x(N-1)]
        if self.cyclic_prefix_length != 0:
            ofdm_sym = concatenate((output_ifft[-self.cyclic_prefix_length:], output_ifft))
        else:
            ofdm_sym = output_ifft

        return ofdm_sym

    def demodulate(self, input_symbols):
        # discard cyclic prefix
        input_fft = input_symbols[self.cyclic_prefix_length:]

        if len(input_fft) != self.fft_size:
            raise ValueError("Length of FFT input should be equal to the FFT size")

        # fast fourier transform
        output_fft = fft(input_fft, self.fft_size)

        # collect data symbols from data subcarriers
        data_symbols = zeros(len(self.data_subcarriers),dtype='complex')
        for i in range(len(data_symbols)):
            data_symbols[i] = output_fft[self.data_subcarriers[i]]

        return data_symbols

    @property
    def data_subcarriers(self):
        return self._data_subcarriers

    @data_subcarriers.setter
    def data_subcarriers(self, data_subcarriers):
        test_input_pass = True
        if not isinstance(data_subcarriers,ndarray):
            test_input_pass = False
        if data_subcarriers.ndim() > 1:
            test_input_pass = False
        if not test_input_pass:
            raise ValueError("data subcarriers must be 1D ndarray type.")
        self._data_subcarriers = data_subcarriers
        all_subcarriers = arange(self._fft_size)
        self._zero_subcarriers = all_subcarriers[isin(all_subcarriers,data_subcarriers,invert='True')]

    @property
    def zero_subcarriers(self):
        return self._zero_subcarriers

    @zero_subcarriers.setter
    def zero_subcarriers(self, zero_subcarriers):
        test_input_pass = True
        if not isinstance(zero_subcarriers,ndarray):
            test_input_pass = False
        if zero_subcarriers.ndim > 1:
            test_input_pass = False
        if not test_input_pass:
            raise ValueError("zero subcarriers must be 1D ndarray type.")
        self._zero_subcarriers = zero_subcarriers
        all_subcarriers = arange(self._fft_size)
        self._data_subcarriers = all_subcarriers[isin(all_subcarriers,zero_subcarriers,invert='True')]

    @property
    def fft_size(self):
        return self._fft_size

    @fft_size.setter
    def fft_size(self, fft_size):
        log2_fft_size = log2(fft_size)
        if log2_fft_size != int(log2_fft_size):
            raise ValueError("The OFDM model only accepts fft size being power of 2")
        self._fft_size = fft_size
        self._data_subcarriers = arange(fft_size)
        self._zero_subcarriers = array([], dtype='int')


    @property
    def cyclic_prefix_length(self):
        return self._cyclic_prefix_length

    @cyclic_prefix_length.setter
    def cyclic_prefix_length(self, cyclic_prefix_length):
        if cyclic_prefix_length > self.fft_size:
            raise ValueError("cyclic prefix length must be less than the fft size")
        self._cyclic_prefix_length = cyclic_prefix_length


class OfdmModem_80211a(OfdmModem):
    """OFDM Modem for 802.11a standard in 20MHz"""
    def __init__(self):
        self.fft_size = 64
        self.zero_subcarriers = concatenate((array([0]), arange(27,38)))
        # data_subcarriers is automatically updated
        self.cyclic_prefix_length = 16 # T_cyclic_prefix = 0.8us with 50ns sample interval
