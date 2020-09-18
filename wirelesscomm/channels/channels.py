"""
Channel Models
==============

"""

from numpy import sqrt, asarray, convolve, ceil, exp, zeros

from numpy.random import standard_normal

__all__ = ['AWGNChannel','SISOFadingChannel','SISOIndoor80211aChannel']

class _Channel():
    """Base channel model"""

    def __init__(self):
        self.name = 'Base channel'


    def propagate(self, input_frame):
        raise NotImplementedError("Channel Models should implement 'propagate(input_frame)' function")




class AWGNChannel(_Channel):
    """Additive White Gaussian Noise Channel Model"""

    def __init__(self, noise_var=0.0):
        super().__init__()
        self.name = 'AWGN channel'
        self.noise_var = noise_var
        self.noise_std = sqrt(noise_var)
        self.white_gaussian_noises = None
    
    def generate_white_gaussian_noises(self, dimension):
        """
        Generate the white gaussian noises with size according to dimension
        """
        self.white_gaussian_noises = self.noise_std*(standard_normal(size=dimension) +\
                                                     1j*standard_normal(size=dimension))
    
    def set_EbN0_dB(self, ebn0_db, Es=1.0, channel_code_rate=1, bits_per_modulated_symbol=1):
        ebn0_linear = 10**(ebn0_db/10.0)
        self.set_EbN0_linear(ebn0_linear,Es,channel_code_rate,bits_per_modulated_symbol)

    def set_EbN0_linear(self,ebn0_linear, Es=1.0, channel_code_rate=1, bits_per_modulated_symbol=1):
        self.noise_var = Es/(ebn0_linear * channel_code_rate * bits_per_modulated_symbol * 2)
        self.noise_std = sqrt(self.noise_var)
    
    # Maybe should have setter, getter for noise var
    
    def set_noise_power(self, noise_power):
        self.noise_var = noise_power/2.0
        self.noise_std = sqrt(self.noise_var)
    
    def set_noise_var(self, noise_var):
        self.noise_var = noise_var
        self.noise_std = sqrt(self.noise_var)
    
    def propagate(self, input_frame):
        self.generate_white_gaussian_noises(len(input_frame))

        return input_frame + self.white_gaussian_noises



def rayleigh_channel_realization(pdp):
    n_tap = len(pdp)
    h = (standard_normal(n_tap) + 1j*standard_normal(n_tap)) * sqrt(pdp/2.0)
    return h

class SISOFadingChannel(_Channel):
    """ Fading channel
    
    Abbreviation:
        + pdp : power delay profile
    """
    
    def __init__(self, pdp=[1.0], channel_type='constant'):
        self.name = 'SISO Fading channel'
        self.pdp = asarray(pdp)
        if not channel_type in ['constant','rayleigh']:
            raise ValueError("SISO Fading channel accepts only channel type constant or rayleigh")
        self.channel_type = channel_type
        self.h = pdp
    
    
    def set_pdp(self, pdp, channel_type='rayleigh'):
        if not channel_type in ['rayleigh']:
            raise ValueError("Setting PDP with channel type rayleigh only.")
        self.channel_type = channel_type
        self.pdp = pdp
    
    def set_channel_realization(self, h):
        self.channel_type = 'constant'
        self.h = h
    
    def propagate(self, input_frame):
        if self.channel_type == 'rayleigh':
            self.h = rayleigh_channel_realization(self.pdp)
        
        return convolve(input_frame,self.h)
    


class SISOIndoor80211aChannel(SISOFadingChannel):
    """
    802.11a SISO Indoor Channel Model
    
    Input:
        + Trms_ns : RMS delay spread in nanosecond
        + Ts_ns : sampling interval in nanosecond
    """
    
    def __init__(self, Trms_ns=25, Ts_ns=50):
        Ts_Trms_ratio = Ts_ns/Trms_ns
        kmax = int(ceil(10/Ts_Trms_ratio))+1
        pdp = zeros(kmax,dtype='float')
        for i in range(kmax):
            if i == 0:
                pdp[i] = (1 - exp(-Ts_Trms_ratio)) / (1 - exp(-(kmax+1)*Ts_Trms_ratio))
            else:
                pdp[i] = pdp[0]*exp(-i*Ts_Trms_ratio);
        
        super().__init__(pdp,'rayleigh')
        



    
    
    
    
    
    
    