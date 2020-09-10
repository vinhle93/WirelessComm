"""
Tools
"""

from numpy import zeros

__all__ = ['bits_to_symbols', 'symbols_to_bits']

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
