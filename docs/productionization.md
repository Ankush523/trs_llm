# Productionization

Production should serve only frozen `LibrarySnapshot` artifacts.

- offline jobs: trace generation, distillation, index build
- online service: retrieve, gate, prompt, infer, return
- observability: usage, latency, library version, fallback rate
