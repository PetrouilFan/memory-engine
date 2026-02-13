"""
Simple Bloom filter implementation for memory IDs.
This is a lightweight, deterministic filter suitable for pre‑filtering
IDs before the expensive FAISS search. It uses multiple SHA‑256 hashes
to set bits in a fixed‑size bitarray.
"""
import hashlib
import math
from typing import Iterable

class SimpleBloomFilter:
    def __init__(self, capacity: int, error_rate: float = 0.01):
        """Create a Bloom filter.
        capacity: expected number of items.
        error_rate: desired false‑positive probability.
        """
        # Size of bit array (m) and number of hash functions (k)
        self.m = max(1, int(-capacity * math.log(error_rate) / (math.log(2) ** 2)))
        self.k = max(1, int((self.m / capacity) * math.log(2)))
        self.bitarray = bytearray((self.m + 7) // 8)
        self._seeds = [i.to_bytes(4, 'little') for i in range(self.k)]

    def _hashes(self, item: int):
        # Convert numpy types to Python int if needed
        item = int(item)
        b = item.to_bytes(8, 'little', signed=False)
        for seed in self._seeds:
            h = hashlib.sha256(seed + b).digest()
            # Take first 8 bytes for an integer
            yield int.from_bytes(h[:8], 'little') % self.m

    def add(self, item: int):
        for pos in self._hashes(item):
            byte_index = pos // 8
            bit_index = pos % 8
            self.bitarray[byte_index] |= 1 << bit_index

    def __contains__(self, item: int) -> bool:
        return all(
            (self.bitarray[pos // 8] & (1 << (pos % 8))) != 0
            for pos in self._hashes(item)
        )

    def add_all(self, items: Iterable[int]):
        for i in items:
            self.add(i)
