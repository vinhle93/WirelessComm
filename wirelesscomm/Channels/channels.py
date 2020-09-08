"""
Channel Models
==============

"""

from numpy import sqrt

from numpy.random import standard_normal

__all__ = ['AWGNChannel']

class _Channel():
    """Base channel model"""

    def __init__(self):
        self.noise_var = 0.0
        self.noise_std = 0.0
        self.white_gaussian_noises = None
        self.unnoisy_frame = None

    def generate_white_gaussian_noises(self, dimension):
        """
        Generate the white gaussian noises with size according to dimension
        """
        self.white_gaussian_noises = self.noise_std*(standard_normal(size=dimension) + 1j*standard_normal(size=dimension))

    def set_EbN0_dB(self, ebn0_db, Es=1.0, channel_code_rate=1, bits_per_modulated_symbol=1):
        ebn0_linear = 10**(ebn0_db/10.0)
        self.set_EbN0_linear(ebn0_linear,Es,channel_code_rate,bits_per_modulated_symbol)

    def set_EbN0_linear(self,ebn0_linear, Es=1.0, channel_code_rate=1, bits_per_modulated_symbol=1):
        self.noise_var = Es/(ebn0_linear * channel_code_rate * bits_per_modulated_symbol * 2)
        self.noise_std = sqrt(self.noise_var)

    def propagate(self, input_frame):
        raise NotImplementedError("Channel Models should implement 'propagate(input_frame)' function")




class AWGNChannel(_Channel):
    """Additive White Gaussian Noise Channel Model"""

    def __init__(self, noise_var=0.0):
        super().__init__()
        self.noise_var = noise_var
        self.noise_std = sqrt(noise_var)

    def propagate(self, input_frame):
        self.unnoisy_frame = input_frame
        self.generate_white_gaussian_noises(len(input_frame))

        return self.unnoisy_frame + self.white_gaussian_noises
