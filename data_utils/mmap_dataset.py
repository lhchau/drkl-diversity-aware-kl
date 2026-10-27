"""Memory-mapped token store (the fairseq/Megatron "MMIDIDX" format used by MiniLLM).

`<prefix>.bin` holds the concatenated token ids, `<prefix>.idx` their sizes and
byte offsets. Files written by older versions of this code read unchanged.
"""
import struct

import numpy as np

_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64,
           6: np.float32, 7: np.float64, 8: np.uint16, 9: np.uint32}
_CODES = {v: k for k, v in _DTYPES.items()}


class MMapDatasetBuilder:
    def __init__(self, prefix, dtype):
        self._prefix = prefix
        self._dtype = dtype
        self._bin = open(prefix + ".bin", "wb")
        self._sizes = []

    def add_item(self, ids):
        # -1 (the prompt/response separator) wraps to the dtype's max value
        arr = np.array(ids, dtype=np.int64).astype(self._dtype)
        self._bin.write(arr.tobytes(order="C"))
        self._sizes.append(arr.size)

    def finalize(self):
        self._bin.close()
        sizes = np.array(self._sizes, dtype=np.int32)
        pointers = np.zeros(len(sizes), dtype=np.int64)
        pointers[1:] = np.cumsum(sizes[:-1], dtype=np.int64) * np.dtype(self._dtype).itemsize
        with open(self._prefix + ".idx", "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", _CODES[self._dtype]))
            f.write(struct.pack("<Q", len(sizes)))
            f.write(struct.pack("<Q", 1))  # document index, kept for format compatibility
            f.write(sizes.tobytes(order="C"))
            f.write(pointers.tobytes(order="C"))
            f.write(np.zeros(1, dtype=np.int64).tobytes(order="C"))


class MMapDataset:
    def __init__(self, prefix):
        with open(prefix + ".idx", "rb") as f:
            assert f.read(9) == _MAGIC, f"{prefix}.idx is not an MMIDIDX index"
            assert struct.unpack("<Q", f.read(8)) == (1,)
            self.dtype = _DTYPES[struct.unpack("<B", f.read(1))[0]]
            n = struct.unpack("<Q", f.read(8))[0]
            f.read(8)  # document count, unused
            offset = f.tell()
        idx = np.memmap(prefix + ".idx", mode="r", order="C")
        self.sizes = np.frombuffer(idx, dtype=np.int32, count=n, offset=offset)
        self._pointers = np.frombuffer(idx, dtype=np.int64, count=n, offset=offset + self.sizes.nbytes)
        self._bin = np.memmap(prefix + ".bin", mode="r", order="C")

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, i):
        return np.frombuffer(self._bin, dtype=self.dtype, count=self.sizes[i], offset=self._pointers[i])
