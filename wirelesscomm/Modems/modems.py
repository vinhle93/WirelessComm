"""
Modulator-Demodulator Models
============================

"""

from numpy import array, sqrt, zeros, argmin

__all__ = ['bits_to_symbols', 'symbols_to_bits', 'QPSKModem']

def bits_to_symbols(input_bits, bits_per_symbol=1):
    """Convert array of bits into array of symbols, with (bits_per_symbol) bits per symbol"""
    assert len(input_bits)%bits_per_symbol == 0, "input_bits length should be divisible by number of bits per symbol"
    no_symbols = int(len(input_bits)/bits_per_symbol)
    output_symbols = zeros(no_symbols, dtype='int')
    for i in range(no_symbols):
        for j in range(bits_per_symbol):
            output_symbols[i] = output_symbols[i] + (input_bits[i*bits_per_symbol+j] << (bits_per_symbol-j-1))
    return output_symbols

def symbols_to_bits(input_symbols, bits_per_symbol=1):
    """Convert array of symbols into array of bits, with (bits_per_symbol) bits per symbol"""
    no_symbols = len(input_symbols)
    output_bits = zeros(bits_per_symbol*no_symbols, dtype='int')
    for i in range(no_symbols):
        sym = input_symbols[i]
        for j in range(bits_per_symbol-1,-1,-1):
            output_bits[i*bits_per_symbol+j] = sym & 1
            sym = sym >> 1
    return output_bits

class _Modem():
    """Base class of Modulator-Demodulator Models"""

    def __init__(self):
        self.constellation = None
        self.bits_per_modulated_symbol = 0
        raise NotImplementedError("Modems should have a constructor")

    def modulate(self, input_bits):
        input_symbols = bits_to_symbols(input_bits, self.bits_per_modulated_symbol)
        output_symbols = array([self._constellation[symbol] for symbol in input_symbols])
        return output_symbols

    def demodulate(self, input_symbols, noise_var=1e-4, demod_type='soft'):
        if demod_type == 'hard':
            decoded_symbols = array([argmin(abs(sym - self._constellation)) for sym in input_symbols])
            return symbols_to_bits(decoded_symbols, self.bits_per_modulated_symbol)
        elif demod_type == 'soft':
            return self.demodulate_soft()

    def demodulate_soft(self,input_symbols,noise_var):
        raise NotImplementedError("Modems should implement 'demodulate_soft(input_symbols,noise_var)' function")

    @property
    def constellation(self):
        """Return the constellation of the Modem"""
        return self._constellation

    @constellation.setter
    def constellation(self, imported_constellation):
        if (1 << self.bits_per_modulated_symbol) != len(imported_constellation):
            raise ValueError("Imported constellation length must be equal to 2**(bits_per_modulated_symbol)")
        raise ValueError("Currently cannot import constellations")




class QPSKModem(_Modem):
    """QPSK Modulator-Demodulator"""
    def __init__(self):
        self._constellation = sqrt(0.5)*array([-1-1j,-1+1j,1-1j,1+1j])
        self.bits_per_modulated_symbol = 2

    def demodulate_soft(self, input_symbols, noise_var):
        no_symbols = len(input_symbols)
        llr = zeros(self.bits_per_modulated_symbol*no_symbols, dtype='float')
        for i in range(no_symbols):
            llr[2*i] = sqrt(2)*input_symbols[i].real/noise_var
            llr[2*i+1] = sqrt(2)*input_symbols[i].imag/noise_var
        return llr
