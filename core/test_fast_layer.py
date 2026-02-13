import numpy as np
from mem_fast_layer import MemoryLayer

def test_layer():
    layer = MemoryLayer()
    
    # Mock data
    vec = np.random.rand(384).tolist()
    text = "This is a urgent fast memory test"
    
    print("Testing add_to_fast...")
    fid = layer.add_to_fast(memory_id=99999, vector=vec, text=text, importance=5.0)
    print(f"Added to fast layer with ID: {fid}")
    
    print("\nTesting dual_search...")
    results = layer.dual_search(vec, k_fast=3, k_slow=3)
    for r in results:
        print(f"[{r['layer']}] ID {r['memory_id']} - Score: {r['score']:.4f} - Text: {r['text']}")

    print("\nTesting maintenance...")
    layer.maintenance()
    print("Maintenance done.")

if __name__ == "__main__":
    test_layer()
