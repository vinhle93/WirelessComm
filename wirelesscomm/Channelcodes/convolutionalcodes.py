"""
Convolutional encoders and decoders
===================================


"""

from numpy import zeros, flip, argmax

from wirelesscomm.channelcodes.trellis import Trellis
from wirelesscomm.tools import symbols_to_bits, bits_to_symbols

__all__ = ['ConvolutionalCode']

class ConvolutionalCode():
    """
    Convolutional codes, support temporarily only code rate 1/n (w/o puncturing).
    """
    def __init__(self,constraint_length,gen_matrix,polynomial_type='oct',\
                    code_type='nrnsc', termination_type='zero'):
        self.trellis = Trellis(constraint_length,gen_matrix,polynomial_type,code_type)
        self.constraint_length = constraint_length
        self.nu = constraint_length - 1
        self.number_of_states = (1 << self.nu)
        self.gen_matrix = (gen_matrix,polynomial_type)
        self.code = (constraint_length,gen_matrix)
        self.code_type = code_type
        self.no_input_bits = self.trellis.k
        self.no_output_bits = self.trellis.n
        self.code_rate = self.trellis.rate
        self.termination_type = termination_type
        # Add puncturing patterns, modify code_rate


    def encode(self, input_bits):
        input_symbols = bits_to_symbols(input_bits,self.no_input_bits)
        if self.termination_type == 'zero':
            encoded_symbols = zeros(len(input_symbols) + self.trellis.number_of_zero_symbols, dtype='int')
            self.state_sequence = zeros(len(input_symbols) + self.trellis.number_of_zero_symbols + 1, dtype='int')
        else: # termination_type = 'circular'
            encoded_symbols = zeros(len(input_symbols), dtype='int')

        state = 0
        for i in range(len(input_symbols)):
            encoded_symbols[i] = self.trellis.next_symbols[state][input_symbols[i]]
            state = self.trellis.next_states[state][input_symbols[i]]
            self.state_sequence[i+1] = state

        if self.termination_type == 'zero':
            # Pad 0 to force the trellis back to state 0
            for i in range(len(input_symbols),len(input_symbols)+self.trellis.number_of_zero_symbols):
                encoded_symbols[i] = self.trellis.next_symbols[state][0]
                state = self.trellis.next_states[state][0]
                self.state_sequence[i+1] = state
        else: # termination_type = 'circular'
            raise NotImplementedError("encoding for circular termination is not implemented yet.")

        return symbols_to_bits(encoded_symbols,self.no_output_bits)

    def _branch_metric_unit(self, bmu_llrs):
        self.branch_metrics[:] = 0
        for i in range(len(bmu_llrs)):
            position = 1 << i
            for j in range(len(self.branch_metrics)):
                if j & position != 0:
                    self.branch_metrics[j] = self.branch_metrics[j] + bmu_llrs[i]

    def _viterbi_add_compare_select(self):
        next_path_metric = zeros(self.number_of_states, dtype='float')
        next_path_memory = zeros((self.number_of_states,self.traceback_length), dtype='int')
        candidate_metrics = zeros(self.trellis.number_of_branches_per_state, dtype='float')
        for s in range(self.number_of_states):
            for i in range(self.trellis.number_of_branches_per_state):
                candidate_metrics[i] = self.path_metrics[self.trellis.prev_states[s][i]] + \
                                        self.branch_metrics[self.trellis.prev_symbols[s][i]]
            surviving_candidate = argmax(candidate_metrics)
            surviving_state = self.trellis.prev_states[s][surviving_candidate]
            next_path_metric[s] = candidate_metrics[surviving_candidate]
            next_path_memory[s][1:] = self.path_memory[surviving_state][0:-1]
            next_path_memory[s][0] = s

        self.path_metrics = next_path_metric
        self.path_memory = next_path_memory

    def _viterbi_traceback_nrnscc(self, trellis_k, no_trellis_section):
        """
        Abbreviation: ml = maximum likelihood
        This function is currently written for nrnsc code with rate 1/n (w/o puncturing)
        If different code rates are desired, consider change the function.
        """
        if trellis_k == no_trellis_section-1:
            # if reach the end of the trellis, the state 0 is the ml state
            ml_path = self.path_memory[0]
            for i in range(min(self.traceback_length,no_trellis_section)):
                if ml_path[i] < (self.number_of_states >> 1):
                    self.decoded_symbols[no_trellis_section-i-1] = 0
                else:
                    self.decoded_symbols[no_trellis_section-i-1] = 1
        elif trellis_k < self.traceback_length-1:
            # the traceback only active when k > traceback_length
            # or when k reach the end of the trellis.
            pass
        else:
            # Given a set of path metrics, the ml state is the state with
            # the highest path metric.
            # The ml path is then the state sequence ending with the ml state.
            # By traceback on the ml path, the state at position (k - traceback_length + 1)
            # is obtained, hence, the symbol decision.
            ml_state = argmax(self.path_metrics)
            ml_path = self.path_memory[ml_state]
            if ml_path[-1] < (self.number_of_states >> 1):
                self.decoded_symbols[trellis_k - self.traceback_length + 1] = 0
            else:
                self.decoded_symbols[trellis_k - self.traceback_length + 1] = 1

    def _viterbi_traceback_rsc(self, trellis_k, no_trellis_section):
        raise NotImplementedError("Viterbi traceback for RSC code is not implemented yet.")

    def viterbi_decode(self, bit_llrs):
        self.branch_metrics = zeros(1 << self.no_output_bits, dtype='float')
        self.traceback_length = 2 * self.nu
        self.path_memory = zeros((self.number_of_states,self.traceback_length), dtype='int')
        self.path_metrics = zeros(self.number_of_states, dtype='float')
        if self.termination_type == 'zero':
            self.path_metrics[0] = 10.0

        no_trellis_sections = int(len(bit_llrs)/self.no_output_bits)
        self.decoded_symbols = zeros(no_trellis_sections, dtype='int')
        section_llrs = bit_llrs.reshape((no_trellis_sections,self.no_output_bits))

        if self.code_type == 'nrnsc':
            viterbi_traceback = self._viterbi_traceback_nrnscc
        else:
            viterbi_traceback = self._viterbi_traceback_rsc

        for k in range(no_trellis_sections):
            self._branch_metric_unit(flip(section_llrs[k]))

            self._viterbi_add_compare_select()

            viterbi_traceback(k, no_trellis_sections)

        if self.code_type == 'nrnsc':
            return symbols_to_bits(self.decoded_symbols[:-self.trellis.number_of_zero_symbols], self.no_input_bits)
        else:
            return symbols_to_bits(self.decoded_symbols, self.no_input_bits)
