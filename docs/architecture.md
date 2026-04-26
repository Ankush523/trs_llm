# Architecture

The TRS pipeline has four main stages:

1. dataset preparation
2. direct trace generation
3. skill distillation and library build
4. direct or TRS evaluation plus comparison

Runtime requests use a frozen library snapshot and never read raw traces.
