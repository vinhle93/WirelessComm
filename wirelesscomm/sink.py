"""
==========
Sink Model
==========
"""

import wirelesscomm.source

__all__ = ['Sink']

class Sink():
    """
    A sink linked to a source compares decoded data with generated data
    and produces Bit Error Rate, Frame Error Rate

    """
    def __init__(self, source):
        self.source = source
        self.frame_size = source.get_frame_size()
        self.source.set_sink(self)
        self.decoded_frame = None
        self.origin_frame = None
        self.error_meter_activated = True
        self.number_of_bit_errors = 0
        self.number_of_frame_errors = 0
        self.number_of_frames = 0

    def receive_origin_frame(self,frame):
        self.origin_frame = frame

    def receive_decoded_frame(self,frame):
        if len(frame) < self.frame_size:
            raise ValueError("Decoded frame size exceeds origin frame size")
        elif len(frame) > self.frame_size:
            self.decoded_frame = frame[0:self.frame_size]
        else:
            self.decoded_frame = frame
        if self.error_meter_activated:
            self.number_of_frames = self.number_of_frames + 1
            current_bit_errors = sum(self.origin_frame != self.decoded_frame)
            if current_bit_errors > 0:
                self.number_of_bit_errors = self.number_of_bit_errors + current_bit_errors
                self.number_of_frame_errors = self.number_of_frame_errors + 1

    def deactivate_error_meter(self):
        self.error_meter_activated = False

    def activate_error_meter(self):
        self.error_meter_activated = True

    def get_frame_size(self):
        return self.frame_size

    def get_origin_frame(self):
        return self.origin_frame

    def get_decoded_frame(self):
        return self.decoded_frame

    def get_error_rate(self):
        bit_error_rate = self.number_of_bit_errors/(self.number_of_frames * self.frame_size)
        frame_error_rate = self.number_of_frame_errors/self.number_of_frames
        return (bit_error_rate, frame_error_rate)

    def reset_error_meter(self):
        self.number_of_bit_errors = 0
        self.number_of_frame_errors = 0
        self.number_of_frames = 0
