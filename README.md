# Collaborative GPU Preemption via Spatial Multitasking for Efficient GPU Sharing


To report issues: jizr@connect.hku.hk


CPSpatial executes on AMD GCN 5th (Vega) architecture GPUs, such as Vega56, Vega64, and  Radeon VII (It is only tested on AMD Radeon VII).  However, it does not mean that CPSpatial is limited to this GPU architecture. CPSpatial can be easily ported to other GPU platforms by changing the instruction set architecture.


## Hardware
AMD GCN 5th (Vega) architecture GPUs, such as Vega56, Vega64, and  Radeon VII

Remainder: Multi-GPU causes some bugs in some versions of AMD ROCm. It is better to only plug a single GPU.

## Dependencies
1. Ubuntu Desktop 18.04
2. AMD ROCm v3.7.0
3. g++ v9.3.0 (C++17)

## Compilation
```
cd cpspatial
mkdir build && cd build
cmake .. & make -j
```

## Run all test cases

```
./cpspatial -fullexp
```


## Customized test cases

```
./cpspatial -interval AVG_LS_JOB_ARRIVAL_INTERVAL
```