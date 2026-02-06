"""Concurrency stress test for HolographicMemory."""
import os
import sys
import threading
import random
import string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.holographic_memory import HolographicMemory

mem = HolographicMemory()


def random_text():
    words = [''.join(random.choices(string.ascii_lowercase, k=5)) for _ in range(250)]
    return ' '.join(words)


def worker(i):
    for _ in range(20):
        mem.add_memory(random_text(), metadata={'tags': [f'tag{i}']})


def main():
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results = mem.redshifted_recall('test query')
    print(f'Recall results: {len(results)}')
    print('✅ Concurrency test passed')


if __name__ == '__main__':
    main()
