"""
Trellis of convolutional codes
"""

from numpy import zeros, ndarray, array, asarray

from wirelesscomm.tools import symbols_to_bits, bits_to_symbols

__all__ = ['convert_decimals_to_bins', 'convert_octals_to_bins', 'Trellis']

def convert_decimals_to_bins(decs, binwidth):
    bins_out = symbols_to_bits(decs,binwidth)
    bins_out = bins_out.reshape((len(decs),binwidth))
    return bins_out

def convert_octals_to_bins(octals, binwidth):
    decimals = zeros(len(octals), dtype = 'int')
    for i in range(len(octals)):
        decimals[i] = int('0'+str(octals[i]),8)
    return convert_decimals_to_bins(decimals,binwidth)


class Trellis():
    """
    Trellis of convolutional codes.
    The Trellis class holds a copy of a trellis section.
    The trellis diagram of the code is a cascade of identical trellis sections.

    Abbreviation:
        + rsc : recursive systematic convolutional
        + nrnsc : non-recursive non-systematic convolutional

    Temporary restricted to convolutional codes:
        + rate 1/n for nrnsc codes
        + rate (n-1)/n for rsc codes

    Input:
        + constraint_length : int
                              Number of memory elements + 1

        + gen_matrix        : 1D ndarray
                              (1 x n) polynomial generator matrix

        + polynomial_type   : ['oct', 'dec'], default = 'oct'
                              representation of gen_matrix elements
                              Ex: 133 with 'oct' = 1011011 (constraint_length = 7)
                              while 133 with 'dec' = 10000101 (constrait_length = 8)

        + code_type         : ['nrnsc', 'rsc'], default = 'nrnsc'
                              type of convolutional codes
                              Given (1 x n) gen matrix, the convention is
                              if 'nrnsc' code => code rate = 1/n
                              if 'rsc' code   => code rate = (n-1)/n
                                with the first (n-1) polynomials is the numerators
                                and the last polynomial is the denumerator
    """
    def __init__(self, constraint_length, gen_matrix, polynomial_type='oct', code_type='nrnsc'):
        generator_matrix = asarray(gen_matrix)
        if not self._passed_generator_matrix_check(generator_matrix):
            raise ValueError("Only 1 x n ndarray generator matrices accepted.")
        if polynomial_type == 'oct':
            self.gen_matrix = convert_octals_to_bins(generator_matrix,constraint_length)
        elif polynomial_type == 'dec':
            self.gen_matrix = convert_decimals_to_bins(generator_matrix,constraint_length)
        else:
            raise ValueError("polynomial_type can only be either 'oct' or 'dec'.")

        if not code_type in ['rsc','nrnsc']:
            raise ValueError("code_type can only be either 'rsc' or 'nrnsc'")
        self.code_type = code_type

        self.constraint_length = constraint_length
        self.nu = constraint_length - 1
        self.number_of_states = (1 << self.nu)
        self.n = len(gen_matrix)    # number of output bits per trellis section
        # k: number of input bits per trellis section
        if self.code_type == 'nrnsc':
            self.k = 1
        else:
            self.k = self.n - 1
        self.rate = self.k/self.n
        self.number_of_branches_per_state = 1 << self.k
        self.next_states = zeros((self.number_of_states, self.number_of_branches_per_state), dtype='int')
        self.prev_states = zeros((self.number_of_states, self.number_of_branches_per_state), dtype='int')
        self.next_symbols = zeros((self.number_of_states, self.number_of_branches_per_state), dtype='int')
        self.prev_symbols = zeros((self.number_of_states, self.number_of_branches_per_state), dtype='int')
        self._construct_trellis()


    def _passed_generator_matrix_check(self, gen_matrix):
        if not isinstance(gen_matrix, ndarray):
            return False
        if gen_matrix.ndim > 1:
            return False
        if len(gen_matrix) < 2:
            return False
        return True

    def _construct_trellis(self):
        if self.code_type == 'nrnsc':
            self._construct_nrnsc_trellis()
        else:
            self._construct_rsc_trellis()

    def _construct_nrnsc_trellis(self):
        output_bits = zeros(self.n, dtype='int')
        for state in range(self.number_of_states):
            bin_state = symbols_to_bits(array([state]),self.nu)
            for u in range(self.number_of_branches_per_state):
                next_state = (state >> 1) + (u << self.nu-1) # Change here when generalize from 1/2 to k/n
                for i in range(self.n):
                    output_bits[i] = (self.gen_matrix[i][0] & u + sum(self.gen_matrix[i][1:] & bin_state))%2
                output_symbol = bits_to_symbols(output_bits,self.n)[0]

                self.next_states[state][u] = next_state
                self.next_symbols[state][u] = output_symbol

                self.prev_states[next_state][state%2] = state
                self.prev_symbols[next_state][state%2] = output_symbol

        # Get the number of zero symbols to make trellis to zero
        self.number_of_zero_symbols = self.nu

    def _construct_rsc_trellis(self):
        raise NotImplementedError("Trellis construnction for RSC codes have not been defined yet!")
