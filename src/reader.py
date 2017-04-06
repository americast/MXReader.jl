import numpy as np
class SimpleIter:
    def __init__(self, data_names, data_shapes, data_gen,
                 label_names, label_shapes, label_gen, num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            assert len(data) > 0, "Empty batch data."
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            assert len(label) > 0, "Empty batch label."
            return SimpleBatch(data, label)
        else:
            raise StopIteration

def read(self):
    """Read a record as string.
    Returns
    ----------
    buf : string
        Buffer read.
    """
    assert not self.writable
    buf = ctypes.c_char_p()
    size = ctypes.c_size_t()
    check_call(_LIB.MXRecordIOReaderReadRecord(self.handle,
                                               ctypes.byref(buf),
                                               ctypes.byref(size)))
    if buf:
        buf = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char*size.value))
        return buf.contents.raw
    else:
return None

#-----
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(self.uri, ctypes.byref(self.handle)))
self.writable = False
#-----

int MXRecordIOReaderCreate(const char *uri,
                           RecordIOHandle *out) {
  API_BEGIN();
  dmlc::Stream *stream = dmlc::Stream::Create(uri, "r");
  MXRecordIOContext *context = new MXRecordIOContext;
  context->reader = new dmlc::RecordIOReader(stream);
  context->writer = NULL;
  context->stream = stream;
  context->read_buff = new std::string();
  *out = reinterpret_cast<RecordIOHandle>(context);
  API_END();
}
#------
