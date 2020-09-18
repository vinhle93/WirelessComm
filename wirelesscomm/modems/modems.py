"""
Modulator-Demodulator Models
============================

"""

from numpy import array, sqrt, zeros, argmin, exp, pi, log
import matplotlib.pyplot as plt

from wirelesscomm.tools import bits_to_symbols, symbols_to_bits


__all__ = ['QPSKModem', 'PSK8Modem']


class _Modem():
    """Base class of Modulator-Demodulator Models"""

    def __init__(self):
        self.constellation = None
        self.bits_per_modulated_symbol = 0
        self.modem_name = ""
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
            return self.demodulate_soft(input_symbols, noise_var)

    def demodulate_soft(self,input_symbols,noise_var):
        raise NotImplementedError("Modems should implement 'demodulate_soft(input_symbols,noise_var)' function")

    def plot_constellation(self):
        plt.figure()
        plt.scatter(self.constellation.real, self.constellation.imag)
        xmax = max(self.constellation.real)
        ymax = max(self.constellation.imag)
        for symbol in self.constellation:
            plt.text(symbol.real, symbol.imag + 0.05, self.demodulate([symbol],demod_type='hard'))
        plt.grid(True, which="both")
        plt.axhline()
        plt.axvline()
        plt.title(self.modem_name + " Constellation")
        plt.xlim(-xmax-0.3,xmax+0.3)
        plt.ylim(-ymax-0.3,ymax+0.3)
        plt.show()

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
        self.modem_name = "QPSK"

    def demodulate_soft(self, input_symbols, noise_var):
        no_symbols = len(input_symbols)
        llr = zeros(self.bits_per_modulated_symbol*no_symbols, dtype='float')
        for i in range(no_symbols):
            llr[2*i] = sqrt(2)*input_symbols[i].real/noise_var
            llr[2*i+1] = sqrt(2)*input_symbols[i].imag/noise_var
        return llr


class PSK8Modem(_Modem):
    """8PSK Modulator-Demodulator with Gray constellation"""
    def __init__(self):
        gray_mapping = array([0,1,3,2,7,6,4,5], dtype='int')
        self._constellation = array([exp(1j*position*pi/4.0) for position in gray_mapping])
        self.bits_per_modulated_symbol = 3
        self.modem_name = "8PSK"

    def demodulate_soft(self, input_symbols, noise_var):
        no_symbols = len(input_symbols)
        llr = zeros(self.bits_per_modulated_symbol*no_symbols, dtype='float')
        for i in range(no_symbols):
            # Pr[x]: probability of receiving input_symbols[i], given constellation[x] transmitted
            Pr = exp(-(abs(input_symbols[i] - self._constellation)**2)/(2*noise_var))
            llr[3*i] = log((Pr[4]+Pr[5]+Pr[6]+Pr[7]) / (Pr[0]+Pr[1]+Pr[2]+Pr[3]))
            llr[3*i+1] = log((Pr[2]+Pr[3]+Pr[6]+Pr[7]) / (Pr[0]+Pr[1]+Pr[4]+Pr[5]))
            llr[3*i+2] = log((Pr[1]+Pr[3]+Pr[5]+Pr[7]) / (Pr[0]+Pr[2]+Pr[4]+Pr[6]))
        return llr
