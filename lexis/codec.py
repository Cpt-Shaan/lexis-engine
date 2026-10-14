"""Integer codecs for postings: variable byte, Elias gamma, and fixed-width bit packing.

Variable byte follows IIR Fig. 5.8: 7 payload bits per byte, and the high
bit is set on the *last* byte of each number.
"""

import numpy as np


# --------------------------------------------------------------------------- VB
def vb_encode_number(n, out):
    if n < 128:
        out.append(n | 128)
        return
    tmp = []
    while True:
        tmp.append(n & 127)
        if n < 128:
            break
        n >>= 7
    tmp.reverse()
    tmp[-1] |= 128
    out.extend(tmp)


def vb_encode(numbers, out=None):
    if out is None:
        out = bytearray()
    for n in numbers:
        if n < 128:
            out.append(n | 128)
        else:
            vb_encode_number(n, out)
    return out


def vb_decode(data, start=0, end=None):
    """Pure-Python VB decode, used for short lists and streaming readers."""
    if end is None:
        end = len(data)
    out = []
    n = 0
    for i in range(start, end):
        b = data[i]
        if b < 128:
            n = (n << 7) | b
        else:
            out.append((n << 7) | (b & 127))
            n = 0
    return out


def vb_decode_np(data):
    """Vectorized VB decode of a whole buffer -> int64 array."""
    a = np.frombuffer(data, dtype=np.uint8)
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    ends = np.flatnonzero(a >= 128)
    starts = np.empty_like(ends)
    starts[0] = 0
    starts[1:] = ends[:-1] + 1
    lengths = ends - starts + 1
    if lengths.max() == 1:
        return (a & 127).astype(np.int64)
    gid = np.repeat(np.arange(ends.size), lengths)
    shift = 7 * (ends[gid] - np.arange(a.size))
    vals = (a & 127).astype(np.int64) << shift
    return np.add.reduceat(vals, starts)


class VBReader:
    """Sequential cursor over a VB byte string."""

    __slots__ = ("data", "pos")

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos

    def next(self):
        data = self.data
        n = 0
        p = self.pos
        while True:
            b = data[p]
            p += 1
            if b < 128:
                n = (n << 7) | b
            else:
                self.pos = p
                return (n << 7) | (b & 127)

    def take(self, k):
        return [self.next() for _ in range(k)]


# ------------------------------------------------------------------------ gamma
def gamma_bits(n):
    """Length in bits of the gamma code of n >= 1."""
    return 2 * (n.bit_length() - 1) + 1


def gamma_encode(numbers):
    """Elias gamma (IIR Sec. 5.3.2): unary(len(offset)) then offset. numbers >= 1."""
    parts = []
    for n in numbers:
        if n < 1:
            raise ValueError("gamma codes need positive integers")
        off = bin(n)[3:]  # binary without the leading 1
        parts.append("1" * len(off) + "0" + off)
    bits = "".join(parts)
    pad = (-len(bits)) % 8
    bits += "1" * pad  # pad with 1s: an incomplete unary prefix is ignored on decode
    return int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b"", len(bits) - pad


def gamma_decode(data, nbits):
    bits = bin(int.from_bytes(data, "big"))[2:].zfill(len(data) * 8)[:nbits]
    out = []
    i = 0
    n = len(bits)
    while i < n:
        j = bits.index("0", i)
        L = j - i
        off = bits[j + 1 : j + 1 + L]
        out.append(int("1" + off, 2))
        i = j + 1 + L
    return out


# ------------------------------------------------------------- fixed bit-packing
def packed_bits(gaps):
    """Fixed-width packing of one list: every gap in ceil(log2(max+1)) bits (+ a 1-byte width header)."""
    if len(gaps) == 0:
        return 8
    w = max(1, int(max(gaps)).bit_length())
    return 8 + w * len(gaps)


# ------------------------------------------------------------------------ gaps
def to_gaps(sorted_ids):
    prev = 0
    out = []
    for x in sorted_ids:
        out.append(x - prev)
        prev = x
    return out


def from_gaps(gaps):
    out = []
    acc = 0
    for g in gaps:
        acc += g
        out.append(acc)
    return out
