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
│  │   OpenCode     │──── HTTP ──→ LiteLLM ──→ vLLM ──→ Model     │
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
│  ┌─────────────────────────────┐    ┌───────────────────────────┐│
│  │  LiteLLM Pod                 │    │  vLLM Pod                 ││
│  │  ┌─────────────────────────┐ │    │                           ││
│  │  │ Token-Cap Proxy :4000   │ │    │  - Qwen 2.5 Coder 7B     ││
│  │  │ (caps max_tokens to fit │─┼───→│  - GPU: nvidia.com/gpu: 1││
│  │  │  available context)     │ │    │  - 20K context window     ││
│  │  └───────────┬─────────────┘ │    │  - Port 8000              ││
│  │              ▼               │    └───────────────────────────┘│
│  │  ┌─────────────────────────┐ │                                 │
│  │  │ LiteLLM :4001           │ │                                 │
│  │  │ - API Gateway           │ │                                 │
│  │  │ - Auth / Rate Limit     │ │                                 │
│  │  └─────────────────────────┘ │                                 │
│  └─────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Why the Token-Cap Proxy?**

Some clients (like OpenCode) hardcode large `max_tokens` values (e.g., 32,000) that exceed the model's context window. The proxy intercepts requests, estimates input token count, and caps `max_tokens` to fit within available context space. This prevents `ContextWindowExceededError` without requiring client-side changes.

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

# Install huggingface CLI (using pipx to avoid system Python conflicts)
pipx install huggingface-hub

# Download Qwen 2.5 Coder 7B AWQ (recommended for 16GB GPUs)
# AWQ quantization reduces model size from ~14GB to ~5GB
hf download Qwen/Qwen2.5-Coder-7B-Instruct-AWQ --local-dir /models/qwen25-coder-7b-awq

# Alternative: Full precision model (requires ~14GB free VRAM, no desktop apps)
# hf download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir /models/qwen-coder-7b

# For larger GPUs (48GB+): Codestral 22B
# NOTE: Requires accepting license terms on huggingface.co first
# hf download mistralai/Codestral-22B-v0.1 --local-dir /models/codestral-22b
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

> **Note**: If you're running a desktop environment (Xorg, Cinnamon, etc.), GUI apps consume GPU memory (often 2-4GB). The settings below are tuned for desktop use with ~12-13GB available VRAM.

```yaml
# Save as: vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen25-coder-7b
  namespace: ai-platform
  labels:
    app: vllm-qwen25-coder-7b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-qwen25-coder-7b
  template:
    metadata:
      labels:
        app: vllm-qwen25-coder-7b
    spec:
      runtimeClassName: nvidia
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.3.post1
          args:
            - "--model"
            - "/models/qwen25-coder-7b-awq"
            - "--served-model-name"
            - "qwen-coder"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "8000"
            - "--max-model-len"
            - "20480"                             # 20K context (adjust based on VRAM)
            - "--quantization"
            - "awq"
            - "--gpu-memory-utilization"
            - "0.80"                              # 80% of available VRAM
            - "--enforce-eager"                   # Disable CUDA graphs (saves memory)
            - "--enable-auto-tool-choice"
            - "--tool-call-parser"
            - "hermes"
            - "--enable-chunked-prefill"          # Better memory efficiency
            - "--max-num-seqs"
            - "16"                                # Limit concurrent requests
            # Uncomment for multi-GPU:
            # - "--tensor-parallel-size"
            # - "2"
          env:
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "max_split_size_mb:512,expandable_segments:True"
          ports:
            - containerPort: 8000
              name: http
          resources:
            limits:
              nvidia.com/gpu: "1"
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
            initialDelaySeconds: 120
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
            sizeLimit: 8Gi                        # Increased for larger context
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen25-coder-7b
  namespace: ai-platform
spec:
  selector:
    app: vllm-qwen25-coder-7b
  ports:
    - port: 8000
      targetPort: 8000
      name: http
  type: ClusterIP
```

**Memory tuning guide:**

| GPU Available VRAM | max-model-len | gpu-memory-utilization |
|--------------------|---------------|------------------------|
| 24GB (dedicated)   | 32768         | 0.90                   |
| 16GB (dedicated)   | 24576         | 0.85                   |
| 12-13GB (desktop)  | 20480         | 0.80                   |
| 10-11GB (desktop)  | 16384         | 0.75                   |

If you get OOM errors, reduce `max-model-len` first, then `gpu-memory-utilization`.

```bash
kubectl apply -f vllm-deployment.yaml

# Watch the pod come up (model loading can take 2-5 minutes)
kubectl -n ai-platform logs -f deployment/vllm-qwen25-coder-7b

# Wait for ready
kubectl -n ai-platform wait --for=condition=available deployment/vllm-qwen25-coder-7b --timeout=300s
```

### 5.4 Verify vLLM is Serving

```bash
# Port-forward to test locally
kubectl -n ai-platform port-forward svc/vllm-qwen25-coder-7b 8000:8000 &

# Test the endpoint
curl http://localhost:8000/v1/models
# Should return: {"data": [{"id": "qwen-coder", ...}]}

# Test a completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder",
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
      - model_name: qwen-coder
        litellm_params:
          model: openai/qwen-coder
          api_base: http://vllm-qwen25-coder-7b.ai-platform.svc.cluster.local:8000/v1
          api_key: "none"                # vLLM doesn't need a key internally

    # If you add more models later, add them here:
    # - model_name: codestral
    #   litellm_params:
    #     model: openai/codestral
    #     api_base: http://vllm-codestral.ai-platform.svc.cluster.local:8000/v1
    #     api_key: "none"

    litellm_settings:
      drop_params: true                  # Ignore unsupported params gracefully
      set_verbose: false
      request_timeout: 120

    general_settings:
      master_key: "sk-your-master-key-change-this"    # CHANGE THIS
      database_url: null                 # Add PostgreSQL later for persistent logs
```

### 6.2 Deploy LiteLLM with Token-Cap Proxy

The deployment includes a Python sidecar proxy that intercepts requests and caps `max_tokens` to fit within the model's context window. This prevents `ContextWindowExceededError` from clients that hardcode large token values.

```yaml
# Save as: litellm-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: token-cap-proxy
  namespace: ai-platform
data:
  proxy.py: |
    #!/usr/bin/env python3
    """Simple proxy that caps max_tokens before forwarding to LiteLLM."""
    import json
    import http.client
    from http.server import HTTPServer, BaseHTTPRequestHandler

    MAX_CONTEXT = 20480   # Must match vLLM's max-model-len
    MAX_OUTPUT = 8192     # Maximum output tokens to allow
    BUFFER = 512          # Safety buffer for tokenization differences
    LITELLM_PORT = 4001

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._proxy_request()

        def do_POST(self):
            self._proxy_request()

        def _proxy_request(self):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''

            if self.path.startswith('/v1/chat/completions') and body:
                body = self._cap_max_tokens(body)

            conn = http.client.HTTPConnection('127.0.0.1', LITELLM_PORT)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ('host', 'content-length')}
            headers['Content-Length'] = str(len(body))

            try:
                conn.request(self.command, self.path, body, headers)
                resp = conn.getresponse()
                self.send_response(resp.status)
                for header, value in resp.getheaders():
                    if header.lower() not in ('transfer-encoding',):
                        self.send_header(header, value)
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except Exception as e:
                self.send_error(502, f'Proxy error: {e}')
            finally:
                conn.close()

        def _cap_max_tokens(self, body):
            try:
                data = json.loads(body)
                input_chars = 0
                for msg in data.get('messages', []):
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        input_chars += len(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                input_chars += len(part.get('text', ''))

                estimated_input = input_chars // 4  # ~4 chars per token
                available = MAX_CONTEXT - estimated_input - BUFFER
                available = max(100, min(available, MAX_OUTPUT))

                requested = data.get('max_tokens', 0)
                if requested > available:
                    print(f'[TokenCap] {requested} -> {available} (input: ~{estimated_input})')
                    data['max_tokens'] = available
                return json.dumps(data).encode()
            except Exception:
                return body

        def log_message(self, format, *args):
            pass  # Suppress default logging

    if __name__ == '__main__':
        server = HTTPServer(('0.0.0.0', 4000), ProxyHandler)
        print('Token-cap proxy listening on :4000 -> LiteLLM :4001')
        server.serve_forever()

---
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
        - name: token-cap-proxy
          image: python:3.11-slim
          command: ["python", "/app/proxy.py"]
          ports:
            - containerPort: 4000
              name: http
          volumeMounts:
            - name: proxy-script
              mountPath: /app
          readinessProbe:
            tcpSocket:
              port: 4000
            initialDelaySeconds: 3
            periodSeconds: 5
          resources:
            requests:
              cpu: "100m"
              memory: "64Mi"
            limits:
              cpu: "500m"
              memory: "256Mi"
        - name: litellm
          image: ghcr.io/berriai/litellm:main-latest
          args:
            - "--config"
            - "/etc/litellm/config.yaml"
            - "--port"
            - "4001"                              # Internal port (proxy on 4000)
            - "--num_workers"
            - "4"
          env:
            - name: LITELLM_MASTER_KEY
              value: "sk-your-master-key-change-this"
          volumeMounts:
            - name: config
              mountPath: /etc/litellm
          readinessProbe:
            httpGet:
              path: /health/liveliness
              port: 4001
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
        - name: proxy-script
          configMap:
            name: token-cap-proxy
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

**How the proxy works:**

```
Client request (max_tokens: 32000)
       │
       ▼
┌─────────────────────────┐
│ Token-Cap Proxy (:4000) │
│ 1. Parse JSON body      │
│ 2. Estimate input tokens│
│ 3. Cap max_tokens to fit│
└───────────┬─────────────┘
            │ (max_tokens: 9724)
            ▼
┌─────────────────────────┐
│ LiteLLM (:4001)         │
│ Routes to vLLM          │
└─────────────────────────┘
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
# LiteLLM is now accessible at http://localhost:30400 (or http://<node-ip>:30400 from other machines)
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

# Test (use localhost if running on the same machine, or node IP from other machines)
export LITELLM_URL="http://localhost:30400"

# List models
curl $LITELLM_URL/v1/models \
  -H "Authorization: Bearer sk-your-master-key-change-this"

# Test chat completion through the gateway
curl $LITELLM_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key-change-this" \
  -d '{
    "model": "qwen-coder",
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
    "models": ["qwen-coder"],
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
# Option A: Quick install script
curl -fsSL https://opencode.ai/install | bash

# Option B: Homebrew (macOS/Linux)
brew install anomalyco/tap/opencode

# Option C: npm
npm i -g opencode-ai@latest

# Verify installation
opencode --version
```

> **Note**: See [github.com/anomalyco/opencode](https://github.com/anomalyco/opencode) for more installation options (Scoop, Chocolatey, Arch, Nix).

### 7.2 Configure OpenCode to Use Your LiteLLM

**Step 1**: Register the LiteLLM provider credentials. Run `opencode`, then inside the TUI:

```
/connect
→ Select "Other"
→ Provider ID: litellm
→ API Key: sk-local-dev
```

This stores credentials in `~/.local/share/opencode/auth.json`.

**Step 2**: Create a configuration file at `~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "litellm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LiteLLM (Local)",
      "options": {
        "baseURL": "http://localhost:30400/v1"
      },
      "models": {
        "qwen-coder": {
          "name": "Qwen 2.5 Coder 7B",
          "context_length": 8192,
          "max_output": 4096
        }
      }
    }
  },
  "model": "litellm/qwen-coder",
  "small_model": "litellm/qwen-coder"
}
```

> **Note**: Replace `localhost:30400` with your node IP if accessing from a different machine.

### 7.3 First Test — Run OpenCode

```bash
# Navigate to any project
cd ~/my-project

# Start OpenCode
opencode

# In the interactive prompt, type:
# > Explain the structure of this project

# OpenCode should:
# 1. Send request to LiteLLM (http://localhost:30400/v1)
# 2. LiteLLM routes to vLLM (qwen-coder)
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
# NAME                                   READY   STATUS    RESTARTS   AGE
# vllm-qwen25-coder-7b-xxxxxxxxx-xxxxx   1/1     Running   0          10m
# litellm-xxxxxxxxx-xxxxx                2/2     Running   0          5m
#                                        ^^^
#                                        2 containers: token-cap-proxy + litellm
```

### 9.2 Test the Full Chain from Terminal

```bash
# Step 1: Verify vLLM directly
kubectl -n ai-platform port-forward svc/vllm-qwen25-coder-7b 8000:8000 &
curl -s http://localhost:8000/v1/models | jq .
kill %1

# Step 2: Verify LiteLLM → vLLM
export LITELLM_URL="http://localhost:30400"
curl -s $LITELLM_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-dev" \
  -d '{
    "model": "qwen-coder",
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
kubectl -n ai-platform logs deployment/vllm-qwen25-coder-7b

# Common issues:
# "CUDA out of memory" → Model too large for GPU. Use quantization or smaller model.
# "Model not found" → Check volume mount path matches where you downloaded weights.
# "Shared memory insufficient" → Increase shm emptyDir sizeLimit.
```

### LiteLLM Can't Reach vLLM

```bash
# Test DNS resolution inside the cluster
kubectl -n ai-platform run debug --rm -it --image=curlimages/curl -- \
  curl http://vllm-qwen25-coder-7b.ai-platform.svc.cluster.local:8000/v1/models

# If DNS fails, check the service exists:
kubectl -n ai-platform get svc vllm-qwen25-coder-7b
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
kubectl -n ai-platform logs -f deployment/vllm-qwen25-coder-7b | grep -i "loading\|ready\|error"
```

### ContextWindowExceededError

**Symptom:**
```
litellm.ContextWindowExceededError: This model's maximum context length is 20480 tokens.
However, you requested 42539 tokens (10539 in the messages, 32000 in the completion).
```

**Cause:** The client (e.g., OpenCode) sends a hardcoded `max_tokens` value (often 32000) that, combined with the input prompt, exceeds vLLM's context window.

**Solutions:**

1. **Use the token-cap proxy** (recommended): The LiteLLM deployment in this guide includes a proxy sidecar that automatically caps `max_tokens` to fit available context. Make sure you're using the deployment from section 6.2.

2. **Increase vLLM context window**: If you have more VRAM available:
   ```yaml
   args:
     - "--max-model-len"
     - "32768"                    # Requires ~14GB free VRAM
     - "--gpu-memory-utilization"
     - "0.90"
   ```

3. **Reduce input size**: If using OpenCode, disable features that add large system prompts, or configure smaller context in `AGENTS.md`.

**Verify the proxy is working:**
```bash
# Check proxy logs
kubectl -n ai-platform logs deployment/litellm -c token-cap-proxy

# Test with a large max_tokens value (should succeed)
curl http://localhost:30400/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-dev" \
  -d '{"model": "qwen-coder", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 32000}'
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
    "title": "Qwen Coder",
    "provider": "openai",
    "model": "qwen-coder",
    "apiBase": "http://<node-ip>:30400/v1",
    "apiKey": "sk-your-developer-key"
  }],
  "tabAutocompleteModel": {
    "title": "Qwen Coder FIM",
    "provider": "openai",
    "model": "qwen-coder",
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
| vLLM logs | `kubectl -n ai-platform logs -f deploy/vllm-qwen25-coder-7b` |
| LiteLLM logs | `kubectl -n ai-platform logs -f deploy/litellm -c litellm` |
| Token-cap proxy logs | `kubectl -n ai-platform logs -f deploy/litellm -c token-cap-proxy` |
| LiteLLM health | `curl http://<node-ip>:30400/health/liveliness` |
| vLLM health | `kubectl -n ai-platform port-forward svc/vllm-qwen25-coder-7b 8000:8000` then `curl localhost:8000/health` |
| List models | `curl -H "Authorization: Bearer <key>" http://<node-ip>:30400/v1/models` |
| Generate API key | `curl -X POST http://<node-ip>:30400/key/generate -H "Authorization: Bearer <master-key>" -d '{"user_id":"alice"}'` |
| Restart vLLM | `kubectl -n ai-platform rollout restart deploy/vllm-qwen25-coder-7b` |
| Restart LiteLLM | `kubectl -n ai-platform rollout restart deploy/litellm` |

---

*Last updated: February 2026*
*Architecture version: 1.1 — Single node K3s with Qwen Coder on vLLM + Token-Cap Proxy*
