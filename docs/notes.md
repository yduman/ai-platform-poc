# Learning Notes

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