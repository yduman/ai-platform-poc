# Learning Notes

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                                 Kubernetes Cluster                             │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                            ai-platform namespace                        │   │
│  │                                                                         │   │
│  │   ┌──────────────────────────────────────────────────────────────────┐  │   │
│  │   │                    vllm-qwen25-coder-7b Pod                      │  │   │
│  │   │                                                                  │  │   │
│  │   │   ┌─────────────────┐      ┌─────────────────┐                   │  │   │
│  │   │   │  vLLM Container │      │   /dev/shm      │                   │  │   │
│  │   │   │                 │      │   (4Gi RAM)     │                   │  │   │
│  │   │   │  --model        │      └─────────────────┘                   │  │   │
│  │   │   │  /models/qwen.. │                                            │  │   │
│  │   │   │                 │                                            │  │   │
│  │   │   │  Port 8000 ─────┼──────────────────────────────────────┐     │  │   │
│  │   │   └────────┬────────┘                                      │     │  │   │
│  │   │            │ mountPath: /models (readOnly)                 │     │  │   │
│  │   └────────────┼───────────────────────────────────────────────┼─────┘  │   │
│  │                │                                               │        │   │
│  │                ▼                                               ▼        │   │
│  │   ┌─────────────────────────┐              ┌─────────────────────────┐  │   │
│  │   │ PVC: model-storage-claim│              │ Service: ClusterIP      │  │   │
│  │   │ requests: 200Gi         │              │ vllm-qwen25-coder-7b    │  │   │
│  │   │ accessMode: ReadOnlyMany│              │ port: 8000              │  │   │
│  │   └───────────┬─────────────┘              └─────────────────────────┘  │   │
│  │               │ bound                                                   │   │
│  └───────────────┼─────────────────────────────────────────────────────────┘   │
│                  │                                                             │
│                  ▼                                                             │
│   ┌──────────────────────────┐                                                 │
│   │ PV: model-storage        │                                                 │
│   │ hostPath: /models        │                                                 │
│   │ capacity: 200Gi          │                                                 │
│   └───────────┬──────────────┘                                                 │
│               │                                                                │
└───────────────┼────────────────────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────┐
│     Host Filesystem           │
│     /models/                  │
│       └── qwen25-coder-7b-awq │
└───────────────────────────────┘
```

## Kubernetes Resources

### PersistentVolume

PersistentVolume (PV) — The actual storage. It's an admin-level resource that says "here's a chunk of storage at this location" (a host directory, NFS share, cloud disk, etc.).

PersistentVolumeClaim (PVC) — A request for storage. It's a pod-level resource that says "I need X amount of storage with Y access mode."

```
PV (admin creates)          PVC (pod requests)           Pod
┌─────────────────┐        ┌─────────────────┐        ┌─────────────┐
│ path: /models   │◄──────►│ needs: 200Gi    │◄───────│ mounts it   │
│ size: 200Gi     │ bound  │ mode: ReadOnly  │  uses  │ at /models  │
└─────────────────┘        └─────────────────┘        └─────────────┘
```

- PV points to `/models` on your host
- PVC is how the vLLM pod requests access to that storage
- Kubernetes binds them together, and the pod mounts it

The separation exists so pods don't need to know where storage actually lives — they just claim what they need, and Kubernetes figures out the binding.

## vLLM

### KV Cache

KV cache (Key-Value cache) stores the computed key and value tensors from the attention mechanism during text generation.

**What it does**

When a transformer generates text token-by-token, each new token needs to "attend" to all previous tokens. Without caching, you'd
recompute the keys and values for every previous token at each generation step - extremely wasteful.

The KV cache stores these computed K and V tensors so they're only calculated once per token.

**Why it grows**

The cache size scales with:
- Sequence length - longer outputs = more cached tokens
- Batch size - more concurrent requests = more caches
- Model dimensions - larger models have bigger K/V tensors
- Number of layers - each layer has its own cache

**Size estimation**

Rough formula per token:
2 × num_layers × hidden_dim × 2 bytes (for fp16/bf16)

For a 7B model (like Llama 2 7B with 32 layers, 4096 hidden dim):
2 × 32 × 4096 × 2 = ~512KB per token

For a 2048 token context: ~1GB per sequence

With multiple concurrent requests or longer contexts (4K, 8K+), this adds up fast.

With a 7B model at bfloat16 (~14GB weights) on a 24GB GPU, you have ~10GB left. But if you want to serve even a few concurrent requests with decent context lengths, the KV cache can easily consume that remaining memory.

**Solutions:**
- Use quantization (4-bit reduces model to ~4GB)
- Reduce `max_model_len` in vLLM
- Use PagedAttention (vLLM does this automatically)
- Reduce `gpu_memory_utilization` less aggressively

