"""

A simple test for the synchronization between source and sink

source generates -> sink obtains the origin frame

"""

from wirelesscomm.source import Source
from wirelesscomm.sink import Sink

my_source = Source(frame_size=8,frame_type=0)

my_sink = Sink(my_source)

source_frame = my_source.generate()

print("Source frame =", source_frame)

print("Origin frame =", my_sink.get_origin_frame())

my_source.set_frame_size(16)

source_frame = my_source.generate()

print("Source frame =", source_frame)

print("Origin frame =", my_sink.get_origin_frame())
