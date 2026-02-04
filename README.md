# On-Prem AI Platform

Learning to create an AI platform environment using Kubernetes, vLLM, and LiteLLM. Run coding agents like OpenCode entirely on your own hardware.

## Architecture

```
OpenCode → Token-Cap Proxy → LiteLLM → vLLM → GPU
           (caps max_tokens)  (gateway)  (inference)
```

## Prerequisites

- **OS**: Linux (Ubuntu 22.04+ recommended)
- **GPU**: NVIDIA with 16GB+ VRAM (RTX 4090, A100, etc.)
- **RAM**: 32GB+
- **Storage**: 100GB+ for model weights
- **Software**:
  - NVIDIA driver 535+
  - NVIDIA Container Toolkit
  - K3s (installed by guide)
  - kubectl, helm

Verify GPU before starting:
```bash
nvidia-smi
```

## Quick Start

```bash
# 1. Install K3s
curl -sfL https://get.k3s.io | sh -
mkdir -p ~/.kube && sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config

# 2. Configure GPU support
sudo nvidia-ctk runtime configure --runtime=containerd \
  --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml
sudo systemctl restart k3s
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# 3. Download model weights
sudo mkdir -p /models && sudo chown $(id -u):$(id -g) /models
pipx install huggingface-hub
hf download Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --local-dir /models/qwen25-coder-7b-awq

# 4. Deploy the stack
kubectl create namespace ai-platform
kubectl apply -f vllm-model-pv.yaml
kubectl apply -f vllm-deployment.yaml
kubectl apply -f litellm-config.yaml
kubectl apply -f litellm-deployment.yaml
kubectl apply -f litellm-nodeport.yaml

# 5. Verify
kubectl -n ai-platform get pods
curl http://localhost:30400/v1/models -H "Authorization: Bearer sk-local-dev"
```

## Usage

**Test the API:**
```bash
curl http://localhost:30400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-dev" \
  -d '{"model": "qwen-coder", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Connect OpenCode:**
```bash
# Install OpenCode
curl -fsSL https://opencode.ai/install | bash

# Configure (~/.config/opencode/opencode.json)
{
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:30400/v1" },
      "models": { "qwen-coder": { "name": "Qwen 2.5 Coder 7B" } }
    }
  },
  "model": "litellm/qwen-coder"
}

# Run
opencode
```

## Configuration

| File | Purpose |
|------|---------|
| `vllm-deployment.yaml` | Model serving (adjust `max-model-len` for context size) |
| `litellm-config.yaml` | API gateway configuration |
| `litellm-deployment.yaml` | Gateway + token-cap proxy |

**Memory tuning** (edit `vllm-deployment.yaml`):

| Available VRAM | max-model-len | gpu-memory-utilization |
|----------------|---------------|------------------------|
| 24GB dedicated | 32768 | 0.90 |
| 16GB dedicated | 24576 | 0.85 |
| 12-13GB desktop | 20480 | 0.80 |

## Documentation

- [Full Bootstrap Guide](docs/on-prem-ai-bootstrap-guide.md) - Complete setup instructions
- [Architecture Notes](docs/notes.md) - Technical details and diagrams

## Troubleshooting

```bash
# Check pod status
kubectl -n ai-platform get pods

# vLLM logs
kubectl -n ai-platform logs -f deploy/vllm-qwen25-coder-7b

# LiteLLM logs
kubectl -n ai-platform logs -f deploy/litellm -c litellm

# Token-cap proxy logs
kubectl -n ai-platform logs -f deploy/litellm -c token-cap-proxy
```

Common issues and solutions are documented in the [troubleshooting section](docs/on-prem-ai-bootstrap-guide.md#10-troubleshooting).
