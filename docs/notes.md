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

### Deployment

Deployment — A controller that manages Pods. Instead of creating Pods directly, you declare "I want N replicas of this Pod template" and the Deployment ensures that state is maintained.

Pod — The smallest deployable unit. A Pod wraps one or more containers that share network/storage and are scheduled together.

```
Deployment                        ReplicaSet                         Pods
┌──────────────────┐            ┌──────────────────┐            ┌─────────────┐
│ replicas: 1      │───creates──│ manages pod      │───creates──│ vllm        │
│ template: {...}  │            │ lifecycle        │            │ container   │
│ strategy: ...    │            │                  │            └─────────────┘
└──────────────────┘            └──────────────────┘
```

- Deployment defines the desired state (image, replicas, resources, probes)
- ReplicaSet is created automatically — it ensures the right number of Pods exist
- If a Pod dies, the ReplicaSet spins up a replacement
- Deployments handle rolling updates: change the image → new ReplicaSet → old Pods gradually replaced

Why not create Pods directly? Pods are mortal — if one crashes or gets evicted, it's gone. Deployments give you self-healing and declarative updates.

### Service

Service — A stable network endpoint for accessing Pods. Pods get random IPs that change when they restart. A Service provides a fixed DNS name and IP that routes to healthy Pods.

```
Client                          Service                           Pods
┌──────────────┐              ┌──────────────────┐            ┌─────────────┐
│ curl         │───request───►│ vllm-qwen25-...  │───routes───│ Pod IP      │
│ svc:8000     │              │ ClusterIP        │    to      │ 10.x.x.x    │
└──────────────┘              │ selector: app=.. │            └─────────────┘
                              └──────────────────┘
```

- Service uses a `selector` to find Pods (e.g., `app: vllm-qwen25-coder-7b`)
- ClusterIP (default) — only accessible within the cluster
- NodePort — exposes on each node's IP at a static port
- LoadBalancer — provisions external load balancer (cloud providers)

The Service acts as an internal load balancer. Even with 1 replica, it provides a stable DNS name (`vllm-qwen25-coder-7b.ai-platform.svc.cluster.local`) so other services don't need to track Pod IPs.

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

