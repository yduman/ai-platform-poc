# On-Prem AI Development Infrastructure: Bootstrap Guide

> **Goal**: Stand up a fully on-prem AI-assisted development environment using K3s, vLLM, LiteLLM, and OpenCode — from zero to a working coding agent on your own hardware.

---

## Table of Contents

1. [Prerequisites & Hardware Requirements](#1-prerequisites--hardware-requirements)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1 — Install K3s](#3-phase-1--install-k3s)
4. [Phase 2 — GPU Support in Kubernetes](#4-phase-2--gpu-support-in-kubernetes)
5. [Phase 3 — Deploy vLLM (Model Serving)](#5-phase-3--deploy-vllm-model-serving)
6. [Phase 4 — Deploy LiteLLM (API Gateway)](#6-phase-4--deploy-litellm-api-gateway)
7. [Phase 5 — Connect OpenCode (Coding Agent)](#7-phase-5--connect-opencode-coding-agent)
8. [Phase 6 — Sandboxing](#8-phase-6--sandboxing)
9. [Verification & End-to-End Test](#9-verification--end-to-end-test)
10. [Troubleshooting](#10-troubleshooting)
11. [Next Steps](#11-next-steps)

---

## 1. Prerequisites & Hardware Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x NVIDIA with 24GB VRAM (e.g., RTX 4090) | 2x NVIDIA A100 80GB or H100 |
| RAM | 32 GB | 64+ GB |
| CPU | 8 cores | 16+ cores |
| Storage | 200 GB SSD (for model weights) | 500 GB+ NVMe |
| Network | Gigabit LAN | 10GbE (if multi-node) |

**Model VRAM requirements (approximate):**

| Model | Parameters | FP16 VRAM | AWQ/GPTQ (4-bit) VRAM |
|-------|-----------|-----------|----------------------|
| Codestral 22B | 22B | ~44 GB | ~12 GB |
| Mistral 3 Large | 123B | ~246 GB | ~65 GB |
| Qwen3 8B | 8B | ~16 GB | ~5 GB |

> **Tip**: If you only have a single 24GB GPU, start with Qwen3 8B (quantized) or Codestral 22B (4-bit quantized) to validate the pipeline. You can always scale later.

### Software

- Linux (Ubuntu 22.04+ or RHEL 8+)
- NVIDIA driver 535+ installed
- NVIDIA Container Toolkit installed
- `curl`, `kubectl`, `helm` available
- Docker or containerd as container runtime

### Verify GPU Before Starting

```bash
# Check NVIDIA driver
nvidia-smi

# Expected output should show your GPU(s), driver version, CUDA version
# If this fails, install drivers first:
# Ubuntu: sudo apt install nvidia-driver-535
# Then reboot
```

---

## 2. Architecture Overview

This is what we are building:

```
┌──────────────────────────────────────────────────────────────────┐
│  Developer Machine                                               │
│  ┌────────────────┐                                              │
│  │   OpenCode     │──── HTTPS ──→ LiteLLM ──→ vLLM ──→ Model   │
│  │   (terminal)   │              (K3s Pod)    (K3s Pod) (GPU)    │
│  └────────────────┘                                              │
│  ┌────────────────┐                                              │
│  │   Sandbox      │ ← Tool execution happens here               │
│  │   (gVisor)     │                                              │
│  └────────────────┘                                              │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  K3s Cluster (single node to start)                              │
│                                                                  │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  LiteLLM Pod         │    │  vLLM Pod                       │ │
│  │  - API Gateway       │───→│  - Codestral 22B               │ │
│  │  - Auth / Rate Limit │    │  - GPU: nvidia.com/gpu: 1      │ │
│  │  - Usage Tracking    │    │  - Port 8000                   │ │
│  │  - Port 4000         │    └─────────────────────────────────┘ │
│  └─────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1 — Install K3s

K3s is a lightweight Kubernetes distribution perfect for on-prem setups.

### 3.1 Install K3s (Single Node)

```bash
# Install K3s with default containerd runtime
curl -sfL https://get.k3s.io | sh -

# Wait for the node to be ready
sudo k3s kubectl get nodes

# Set up kubeconfig for non-root usage
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
export KUBECONFIG=~/.kube/config

# Verify
kubectl get nodes
# NAME         STATUS   ROLES                  AGE   VERSION
# your-host    Ready    control-plane,master   30s   v1.28.x+k3s1
```

### 3.2 Install Helm

```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

### 3.3 Create a Namespace for AI Workloads

```bash
kubectl create namespace ai-platform
```

---

## 4. Phase 2 — GPU Support in Kubernetes

### 4.1 Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 4.2 Configure containerd for NVIDIA (K3s uses containerd)

```bash
# Configure the NVIDIA runtime for containerd
sudo nvidia-ctk runtime configure --runtime=containerd

# K3s uses its own containerd config, so also update that
sudo nvidia-ctk runtime configure --runtime=containerd \
  --config=/var/lib/rancher/k3s/agent/etc/containerd/config.toml

# Restart K3s to pick up changes
sudo systemctl restart k3s
```

### 4.3 Install NVIDIA Device Plugin

```bash
# This allows Kubernetes to discover and schedule GPUs
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml

# Wait for the plugin to be ready
kubectl -n kube-system wait --for=condition=ready pod -l name=nvidia-device-plugin-ds --timeout=60s

# Verify GPU is visible to Kubernetes
kubectl describe node | grep -A 5 "nvidia.com/gpu"
# Should show:
#   nvidia.com/gpu: 1    (or however many GPUs you have)
```

> **Troubleshooting**: If the device plugin logs show `could not load NVML library: libnvidia-ml.so.1: cannot open shared object file`, the plugin needs to run with the NVIDIA runtime. K3s auto-creates an `nvidia` RuntimeClass, but the device plugin doesn't use it by default. Apply this fix:
>
> ```bash
> # Patch the device plugin to use the nvidia RuntimeClass
> kubectl -n kube-system patch daemonset nvidia-device-plugin-daemonset \
>   --type='json' \
>   -p='[{"op": "add", "path": "/spec/template/spec/runtimeClassName", "value": "nvidia"}]'
>
> # Wait for the pod to restart, then verify
> sleep 15
> kubectl describe node | grep -A 5 "nvidia.com/gpu"
> ```

### 4.4 Quick GPU Test

```bash
kubectl run gpu-test --rm -it --restart=Never \
  --image=nvidia/cuda:12.3.1-base-ubuntu22.04 \
  --overrides='{
    "spec": {
      "runtimeClassName": "nvidia",
      "containers": [{
        "name": "gpu-test",
        "image": "nvidia/cuda:12.3.1-base-ubuntu22.04",
        "command": ["nvidia-smi"],
        "resources": {"limits": {"nvidia.com/gpu": "1"}}
      }]
    }
  }' \
  -- nvidia-smi

# Should print the nvidia-smi output and exit
```

---

## 5. Phase 3 — Deploy vLLM (Model Serving)

### 5.1 Download Model Weights (Before Deploying)

Model weights need to be available on the node. For an air-gapped or on-prem environment, download them ahead of time.

```bash
# Create a directory for models
sudo mkdir -p /models
sudo chown $(id -u):$(id -g) /models

# Install huggingface-cli
pip install huggingface_hub

# Download Codestral 22B (or your model of choice)
# NOTE: Some models require accepting license terms on huggingface.co first
huggingface-cli download mistralai/Codestral-22B-v0.1 \
  --local-dir /models/codestral-22b \
  --local-dir-use-symlinks False

# For a smaller test model (if GPU is limited):
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /models/qwen-coder-7b \
  --local-dir-use-symlinks False
```

### 5.2 Create a Persistent Volume for Models

```yaml
# Save as: vllm-model-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-storage
spec:
  capacity:
    storage: 200Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: /models
    type: Directory
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-claim
  namespace: ai-platform
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 200Gi
  volumeName: model-storage
  storageClassName: ""
```

```bash
kubectl apply -f vllm-model-pv.yaml
```

### 5.3 Deploy vLLM

```yaml
# Save as: vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-codestral
  namespace: ai-platform
  labels:
    app: vllm-codestral
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-codestral
  template:
    metadata:
      labels:
        app: vllm-codestral
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "/models/codestral-22b"
            - "--served-model-name"
            - "codestral"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "8000"
            - "--max-model-len"
            - "16384"
            - "--gpu-memory-utilization"
            - "0.90"
            # Uncomment for quantized models:
            # - "--quantization"
            # - "awq"
            # Uncomment for multi-GPU:
            # - "--tensor-parallel-size"
            # - "2"
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              nvidia.com/gpu: 1
          volumeMounts:
            - name: model-volume
              mountPath: /models
              readOnly: true
            - name: shm
              mountPath: /dev/shm
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120     # Models take time to load
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 180
            periodSeconds: 30
      volumes:
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-storage-claim
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 4Gi              # Shared memory for PyTorch
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-codestral
  namespace: ai-platform
spec:
  selector:
    app: vllm-codestral
  ports:
    - port: 8000
      targetPort: 8000
      name: http
  type: ClusterIP
```

```bash
kubectl apply -f vllm-deployment.yaml

# Watch the pod come up (model loading can take 2-5 minutes)
kubectl -n ai-platform logs -f deployment/vllm-codestral

# Wait for ready
kubectl -n ai-platform wait --for=condition=available deployment/vllm-codestral --timeout=300s
```

### 5.4 Verify vLLM is Serving

```bash
# Port-forward to test locally
kubectl -n ai-platform port-forward svc/vllm-codestral 8000:8000 &

# Test the endpoint
curl http://localhost:8000/v1/models
# Should return: {"data": [{"id": "codestral", ...}]}

# Test a completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codestral",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 100
  }'

# Kill the port-forward when done
kill %1
```

---

## 6. Phase 4 — Deploy LiteLLM (API Gateway)

### 6.1 Create LiteLLM Configuration

```yaml
# Save as: litellm-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: litellm-config
  namespace: ai-platform
data:
  config.yaml: |
    model_list:
      - model_name: codestral
        litellm_params:
          model: openai/codestral
          api_base: http://vllm-codestral.ai-platform.svc.cluster.local:8000/v1
          api_key: "none"                # vLLM doesn't need a key internally

    # If you add more models later, add them here:
    # - model_name: qwen-thinking
    #   litellm_params:
    #     model: openai/qwen3-thinking
    #     api_base: http://vllm-qwen.ai-platform.svc.cluster.local:8000/v1
    #     api_key: "none"

    litellm_settings:
      drop_params: true                  # Ignore unsupported params gracefully
      set_verbose: false
      request_timeout: 120

    general_settings:
      master_key: "sk-your-master-key-change-this"    # CHANGE THIS
      database_url: null                 # Add PostgreSQL later for persistent logs
```

### 6.2 Deploy LiteLLM

```yaml
# Save as: litellm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm
  namespace: ai-platform
  labels:
    app: litellm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: litellm
  template:
    metadata:
      labels:
        app: litellm
    spec:
      containers:
        - name: litellm
          image: ghcr.io/berriai/litellm:main-latest
          args:
            - "--config"
            - "/etc/litellm/config.yaml"
            - "--port"
            - "4000"
            - "--num_workers"
            - "4"
          ports:
            - containerPort: 4000
              name: http
          env:
            - name: LITELLM_MASTER_KEY
              value: "sk-your-master-key-change-this"    # CHANGE THIS — match config
          volumeMounts:
            - name: config
              mountPath: /etc/litellm
          readinessProbe:
            httpGet:
              path: /health/liveliness
              port: 4000
            initialDelaySeconds: 10
            periodSeconds: 5
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2"
              memory: "2Gi"
      volumes:
        - name: config
          configMap:
            name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm
  namespace: ai-platform
spec:
  selector:
    app: litellm
  ports:
    - port: 4000
      targetPort: 4000
      name: http
  type: ClusterIP
```

### 6.3 Expose LiteLLM Outside the Cluster

For developers to reach LiteLLM from their machines, you need an Ingress or NodePort.

**Option A: NodePort (simplest for testing)**

```yaml
# Save as: litellm-nodeport.yaml
apiVersion: v1
kind: Service
metadata:
  name: litellm-external
  namespace: ai-platform
spec:
  selector:
    app: litellm
  ports:
    - port: 4000
      targetPort: 4000
      nodePort: 30400
  type: NodePort
```

```bash
kubectl apply -f litellm-nodeport.yaml
# LiteLLM is now accessible at http://<node-ip>:30400
```

**Option B: K3s Traefik Ingress (production-ready)**

```yaml
# Save as: litellm-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: litellm-ingress
  namespace: ai-platform
  annotations:
    traefik.ingress.kubernetes.io/router.tls: "true"
spec:
  rules:
    - host: litellm.your-company.internal
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: litellm
                port:
                  number: 4000
```

### 6.4 Deploy and Verify

```bash
kubectl apply -f litellm-deployment.yaml
kubectl apply -f litellm-nodeport.yaml   # or litellm-ingress.yaml

# Wait for ready
kubectl -n ai-platform wait --for=condition=available deployment/litellm --timeout=60s

# Test (adjust URL based on your exposure method)
export LITELLM_URL="http://<node-ip>:30400"

# List models
curl $LITELLM_URL/v1/models \
  -H "Authorization: Bearer sk-your-master-key-change-this"

# Test chat completion through the gateway
curl $LITELLM_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key-change-this" \
  -d '{
    "model": "codestral",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "max_tokens": 100
  }'
```

### 6.5 Create API Keys for Developers

```bash
# Generate a key for a specific developer
curl $LITELLM_URL/key/generate \
  -H "Authorization: Bearer sk-your-master-key-change-this" \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["codestral"],
    "user_id": "developer-alice",
    "max_budget": 100,
    "metadata": {"team": "backend"}
  }'

# Response includes the key: {"key": "sk-abc123..."}
```

---

## 7. Phase 5 — Connect OpenCode (Coding Agent)

### 7.1 Install OpenCode

```bash
# Install via Go (recommended)
go install github.com/opencode-ai/opencode@latest

# Or download binary from releases
# https://github.com/opencode-ai/opencode/releases
```

### 7.2 Configure OpenCode to Use Your LiteLLM

Create a configuration file at `~/.config/opencode/config.json`:

```json
{
  "provider": {
    "type": "openai",
    "apiKey": "sk-abc123-your-developer-key",
    "baseUrl": "http://<node-ip>:30400/v1"
  },
  "model": {
    "main": "codestral",
    "weak": "codestral"
  }
}
```

Alternatively, use environment variables:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-abc123-your-developer-key"
export OPENAI_BASE_URL="http://<node-ip>:30400/v1"

# Reload
source ~/.bashrc
```

### 7.3 First Test — Run OpenCode

```bash
# Navigate to any project
cd ~/my-project

# Start OpenCode
opencode

# In the interactive prompt, type:
# > Explain the structure of this project

# OpenCode should:
# 1. Send request to LiteLLM (http://<node-ip>:30400/v1)
# 2. LiteLLM routes to vLLM (codestral)
# 3. vLLM runs inference on GPU
# 4. Response streams back through the chain
```

### 7.4 Project-Level Configuration (AGENTS.md)

Create an `AGENTS.md` file in your project root to give OpenCode project-specific context:

```markdown
<!-- Save as: AGENTS.md in your project root -->

# Project: User Authentication Service

## Tech Stack
- Python 3.11, FastAPI, SQLAlchemy, PostgreSQL
- Tests: pytest with pytest-asyncio
- Linting: ruff, mypy

## Conventions
- All endpoints go in `src/routes/`
- Database models in `src/models/`
- Business logic in `src/services/` (never in route handlers)
- Every public function must have a docstring
- Tests mirror source structure: `src/routes/auth.py` → `tests/routes/test_auth.py`

## Commands
- Run tests: `pytest -v`
- Run linting: `ruff check . && mypy src/`
- Start dev server: `uvicorn src.main:app --reload`
```

---

## 8. Phase 6 — Sandboxing

When OpenCode (or any coding agent) executes tools like running shell commands, editing files, or running tests, those actions should be sandboxed.

### 8.1 Option A — gVisor (Recommended for Ease of Setup)

gVisor intercepts system calls and runs them in a userspace kernel, providing container-level isolation.

```bash
# Install gVisor runtime
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | \
  sudo tee /etc/apt/sources.list.d/gvisor.list

sudo apt-get update && sudo apt-get install -y runsc

# Verify
runsc --version
```

**Configure Docker to use gVisor (if using Docker on dev machines):**

```bash
# Add gVisor runtime to Docker
sudo runsc install

# Restart Docker
sudo systemctl restart docker

# Verify
docker run --runtime=runsc hello-world
```

### 8.2 Create a Sandbox Wrapper Script

This script wraps tool execution in a sandboxed container. OpenCode (and similar agents) can be configured to use this for shell commands.

```bash
#!/bin/bash
# Save as: /usr/local/bin/sandbox-exec
# Usage: sandbox-exec <project-dir> <command...>

set -euo pipefail

PROJECT_DIR="${1:?Usage: sandbox-exec <project-dir> <command...>}"
shift
COMMAND="$@"

# Timeout: kill after 60 seconds
TIMEOUT=60

# Run in gVisor-sandboxed container
timeout ${TIMEOUT} docker run \
  --runtime=runsc \
  --rm \
  --network=none \
  --memory=1g \
  --cpus=2 \
  --read-only \
  --tmpfs /tmp:size=500m \
  --tmpfs /home:size=500m \
  -v "${PROJECT_DIR}:/workspace:ro" \
  -v "/tmp/sandbox-output:/output:rw" \
  -w /workspace \
  -e HOME=/home \
  python:3.11-slim \
  bash -c "${COMMAND}"
```

```bash
chmod +x /usr/local/bin/sandbox-exec

# Test it
mkdir -p /tmp/sandbox-output
sandbox-exec ~/my-project "ls -la && python --version"

# What's enforced:
# ✗ No network access (--network=none)
# ✗ No writing to project files (mounted read-only)
# ✗ No access to host filesystem outside project
# ✗ Killed after 60 seconds
# ✗ Max 1GB RAM, 2 CPUs
# ✓ Can read project files
# ✓ Can write to /output and /tmp
# ✓ Can run Python/bash
```

### 8.3 Option B — Firecracker (Stronger Isolation)

For environments where container isolation is not sufficient and you need full VM isolation:

```bash
# Download Firecracker
ARCH=$(uname -m)
curl -L https://github.com/firecracker-microvm/firecracker/releases/latest/download/firecracker-v*-${ARCH}.tgz | tar xz
sudo mv release-*/firecracker-v* /usr/local/bin/firecracker

# Firecracker requires KVM
ls /dev/kvm    # Must exist

# For Firecracker, you'll need a rootfs and kernel image
# This is more involved — see: https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md
```

> **Recommendation**: Start with gVisor. It's much simpler to set up and provides strong isolation for coding agent use cases. Move to Firecracker only if your security requirements explicitly demand VM-level isolation.

### 8.4 Sandbox Flow Summary

```
Developer types: "Add authentication endpoint"
         │
         ▼
OpenCode sends prompt → LiteLLM → vLLM → Codestral
         │
         ▼
Codestral responds: tool_call { "bash": "cat src/auth.py" }
         │
         ▼
OpenCode intercepts tool_call
         │
         ▼
sandbox-exec ~/my-project "cat src/auth.py"
         │
         ▼
gVisor container: reads file, returns content
         │
         ▼
OpenCode sends result back → LiteLLM → vLLM → Codestral
         │
         ▼
... (loop continues until task complete)
```

---

## 9. Verification & End-to-End Test

At this point all components should be running. Here is how to verify the full stack.

### 9.1 Check All Pods

```bash
kubectl -n ai-platform get pods
# NAME                              READY   STATUS    RESTARTS   AGE
# vllm-codestral-xxxxxxxxx-xxxxx    1/1     Running   0          10m
# litellm-xxxxxxxxx-xxxxx           1/1     Running   0          5m
```

### 9.2 Test the Full Chain from Terminal

```bash
# Step 1: Verify vLLM directly
kubectl -n ai-platform port-forward svc/vllm-codestral 8000:8000 &
curl -s http://localhost:8000/v1/models | jq .
kill %1

# Step 2: Verify LiteLLM → vLLM
export LITELLM_URL="http://<node-ip>:30400"
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key-change-this" \
  -d '{
    "model": "codestral",
    "messages": [{"role": "user", "content": "Write a Python function that adds two numbers"}],
    "max_tokens": 200
  }' | jq '.choices[0].message.content'

# Step 3: Verify OpenCode → LiteLLM → vLLM
cd /tmp && mkdir test-project && cd test-project
echo "print('hello')" > main.py
opencode "What does main.py do?"
```

### 9.3 Full Scenario Test

```bash
# Create a test project
mkdir -p /tmp/agent-test && cd /tmp/agent-test
git init

cat > main.py << 'EOF'
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
EOF

cat > AGENTS.md << 'EOF'
# Test Project
Simple math utilities. Tests use pytest.
EOF

# Run OpenCode with a real task
opencode "Add a multiply function and write a test for all three functions"

# Expected behavior:
# 1. OpenCode reads main.py (via sandbox)
# 2. Sends context to LiteLLM → vLLM → Codestral
# 3. Codestral plans: add multiply(), create test file
# 4. OpenCode executes file writes (via sandbox)
# 5. Codestral may run tests (via sandbox)
# 6. Files are written to your project
```

---

## 10. Troubleshooting

### vLLM Pod Won't Start

```bash
# Check logs
kubectl -n ai-platform logs deployment/vllm-codestral

# Common issues:
# "CUDA out of memory" → Model too large for GPU. Use quantization or smaller model.
# "Model not found" → Check volume mount path matches where you downloaded weights.
# "Shared memory insufficient" → Increase shm emptyDir sizeLimit.
```

### LiteLLM Can't Reach vLLM

```bash
# Test DNS resolution inside the cluster
kubectl -n ai-platform run debug --rm -it --image=curlimages/curl -- \
  curl http://vllm-codestral.ai-platform.svc.cluster.local:8000/v1/models

# If DNS fails, check the service exists:
kubectl -n ai-platform get svc vllm-codestral
```

### OpenCode Gets Timeout or Connection Refused

```bash
# Verify LiteLLM is reachable from your machine
curl http://<node-ip>:30400/health/liveliness

# Check if the NodePort service exists
kubectl -n ai-platform get svc litellm-external

# Check firewall (if applicable)
sudo ufw status
# May need: sudo ufw allow 30400/tcp
```

### GPU Not Detected by Kubernetes

```bash
# Check NVIDIA device plugin is running
kubectl -n kube-system get pods -l name=nvidia-device-plugin-ds

# Check node capacity
kubectl describe node | grep -A 10 "Capacity:"
# Should include: nvidia.com/gpu: 1

# If missing, restart the NVIDIA device plugin
kubectl -n kube-system delete pod -l name=nvidia-device-plugin-ds
```

### Model Loading is Very Slow

This is normal on first start — large models take 2-5 minutes to load into GPU memory. Once loaded, inference is fast. Check progress in vLLM logs:

```bash
kubectl -n ai-platform logs -f deployment/vllm-codestral | grep -i "loading\|ready\|error"
```

---

## 11. Next Steps

Once this baseline is working, here is what to add next.

### Immediate Improvements

- **Add PostgreSQL for LiteLLM**: Enables persistent usage tracking and API key management. Deploy postgres in the cluster or use an existing instance, then set `database_url` in LiteLLM config.
- **Add more models**: Deploy additional vLLM instances (e.g., one for Qwen3-Thinking for complex reasoning) and register them in LiteLLM config.
- **TLS everywhere**: Use cert-manager with K3s Traefik for HTTPS on all endpoints.
- **Monitoring**: Add Prometheus + Grafana. vLLM exposes metrics at `/metrics`, and LiteLLM has built-in Prometheus support.

### Editor Integration (Continue.dev)

```bash
# Install Continue extension in VS Code
# Configure ~/.continue/config.json to point at your LiteLLM:
{
  "models": [{
    "title": "Codestral",
    "provider": "openai",
    "model": "codestral",
    "apiBase": "http://<node-ip>:30400/v1",
    "apiKey": "sk-your-developer-key"
  }],
  "tabAutocompleteModel": {
    "title": "Codestral FIM",
    "provider": "openai",
    "model": "codestral",
    "apiBase": "http://<node-ip>:30400/v1",
    "apiKey": "sk-your-developer-key"
  }
}
```

### Chat Interface (LibreChat)

Deploy LibreChat in the same K3s cluster, configured to use LiteLLM as its backend. This gives non-developers (PMs, REs) a ChatGPT-like interface.

### Vector Database & RAG

Deploy Qdrant in the cluster and set up a Cron job to sync Confluence/Jira data into vector stores. This enables semantic search over internal documentation.

### MCP Gateway

Set up MCP servers for Jira, Confluence, and GitLab so that coding agents and LibreChat can pull context from internal tools without manual copy-paste.

---

## Quick Reference — Key URLs and Commands

| What | Command/URL |
|------|-------------|
| K3s status | `sudo systemctl status k3s` |
| All AI pods | `kubectl -n ai-platform get pods` |
| vLLM logs | `kubectl -n ai-platform logs -f deploy/vllm-codestral` |
| LiteLLM logs | `kubectl -n ai-platform logs -f deploy/litellm` |
| LiteLLM health | `curl http://<node-ip>:30400/health/liveliness` |
| vLLM health | `kubectl -n ai-platform port-forward svc/vllm-codestral 8000:8000` then `curl localhost:8000/health` |
| List models | `curl -H "Authorization: Bearer <key>" http://<node-ip>:30400/v1/models` |
| Generate API key | `curl -X POST http://<node-ip>:30400/key/generate -H "Authorization: Bearer <master-key>" -d '{"user_id":"alice"}'` |
| Restart vLLM | `kubectl -n ai-platform rollout restart deploy/vllm-codestral` |
| Restart LiteLLM | `kubectl -n ai-platform rollout restart deploy/litellm` |

---

*Last updated: February 2026*
*Architecture version: 1.0 — Single node K3s with Codestral on vLLM*
