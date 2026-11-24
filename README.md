# Sim manipulation

A reproducible study of learned control for a simulated SO-101 arm. Feasibility is not yet measured.

```mermaid
flowchart LR
  E[MuJoCo environment] --> D[Expert demonstrations]
  D --> T[Policy training]
  T --> V[Rollout evaluation]
  V --> R[Measured results]
```

<!-- results:start -->
Not yet measured.
<!-- results:end -->
