## Characterizing the Performance of Accelerated Jetson Edge Devices for Training Deep Learning Models

[ 🌸 paper : [https://doi.org/10.1145/3570604](https://dl.acm.org/doi/10.1145/3570604) ] | DL for [autonomous vehicles](https://arxiv.org/abs/1912.10773), [smart cities](https://ieeexplore.ieee.org/document/8704334) | DNN model training on the edge - [P1](https://ieeexplore.ieee.org/document/8763885/), [P2](https://ieeexplore.ieee.org/document/8778327).

<p class="ex1" align="justify" style="padding: 5px 5px 5px 5px">
Abstract : Deep Neural Networks (DNNs) have had a significant impact on domains like autonomous vehicles and smart cities through low-latency inferencing on edge computing devices close to the data source. However, DNNtraining on the edge is poorly explored. Techniques like federated learning and the growing capacity of GPU-accelerated edge devices like NVIDIA Jetson motivate the need for a holistic characterization of DNN training on the edge. Training DNNs is resource-intensive and can stress an edge’s GPU, CPU, memory and storage capacities. Edge devices also have different resources compared to workstations and servers, such as slower shared memory and diverse storage media. Here, we perform a principled study of DNN training on individual devices of three contemporary Jetson device types: AGX Xavier, Xavier NX and Nano for three diverse DNN model–dataset combinations. We vary device and training parameters such as I/O pipelining and parallelism, storage media, mini-batch sizes and power modes, and examine their effect on CPU and GPU utilization, fetch stalls, training time, energy usage, and variability. Our analysis exposes several resource inter-dependencies and counter-intuitive insights, while also helping quantify known wisdom. Our rigorous study can help tune the training performance on the edge, trade-off time and energy usage on constrained devices, and even select an ideal edge hardware for a DNN workload, and, in future, extend to federated learning too. As an illustration, we use these results to build a simple model to predict the training time and energy per epoch for any given DNN across different power modes, with minimal additional profiling.
</p>

Federated & Geo Distributed Learning [[paper](https://arxiv.org/abs/1905.10083)], Flower: A Friendly Federated Learning Research Framework [[paper](https://arxiv.org/abs/2007.14390)]

`Study` : Impact of hardware resources such as the number of CPU cores, CPU/GPU/memory frequency, storage media and power modes on the training time and energy usage. Examine how PyTorch settings such as the number of concurrent data loaders, and the DNN model and data  sizes, affect the performance. 

`Takeaways` : Purchasing a faster and more expensive storage may not necessarily improve the training speed if pipelining and caching are able to hide the GPU stalls; a slower and  cheaper hard disk could give the same performance. Similarly, a power mode with the highest GPU  frequency but a lower CPU frequency may not give benefits for smaller DNN models like LeNet which are CPU bound due to pre-processing costs.

 + `(1)` We understand the effect of disk caching, pipelining and parallelizing data fetch and pre
processing on the stall time and epoch training time, and the interplay between CPU and
 GPU performance.
 + `(2)` We study the impact of storage medium and mini-batch sizes on stalls, GPU compute time
 and end-to-end times, and confirm the deterministic performance of these
 devices across time and instances when training.
 + `(3)` We investigate the consequence of Dynamic Voltage and Frequency Scaling (DVFS) and various
 power modes on the training time, energy usage and their trade-off.
 + `(4)` Lastly, we use these results to train simple models to predict the epoch training time and the energy usage per epoch of a given DNN for any power mode with limited profiling.

NVIDIA AGX Orin is comparable to a RTX 3080 Ti workstation GPU, but with a power consumption of ≤ 60𝑊 and no larger than a paperback novel. LPDDR RAM shared between CPU and GPU of AGX Orin vs GDDR RAM in regular GPUs. Edge devices offer several inbuilt and user-defined power modes, each with different cores, CPUfrequency, GPU frequency and memory frequency. This offers a large parameter space (> 29𝑘 combinations for AGX) with interesting power–performance trade-offs. They support a wide variety of storage media including eMMC, Micro SD card, NVME  Solid State Drive (SSD), Hard Disk Drive (HDD), which have different I/O performance and monetary costs.

This study enables accurate modeling of DNN training time and energy usage for the diverse power modes of these devices. This is key for federated learning when devices in a training round need to complete at about the same time.

`DNN Training`: DNN training happens iteratively. In each iteration, we fetch a “mini-batch” of samples from disk to memory, and perform pre-processing on the mini-batch, such as deserialization, cropping, resize, flipping and normalization of the input images using the CPU. Then, the CPU launches kernels on the GPU to perform the forward and backward passes of training. This repeats for the next mini-batch and so on until all the input samples are consumed. This forms one epoch of training. Epochs are repeated using different mini-batch samplings till the model converges. Fetching a mini-batch from disk is I/O intensive, pre-processing is CPU intensive and the training is GPUintensive.

Performing these three stages sequentially will cause the GPU to remain idle while it waits for the disk and CPU to finish fetching and pre-processing a mini-batch. PyTorch’s `DataLoader` and input pipelines constructed from TensorFlow’s `tf.data` API help pipeline the fetch and pre-process stages with the compute stage. The fetch and pre-process stages can also be parallelized to operate on multiple mini-batches so that a fast GPU does not have to wait, or “stall”, for a mini-batch to be ready. 

<img src="./img/jetson.png" width=100%>

All devices of the AGX, NX and Nano run Linux for Tegra (L4T) 𝑣32.5.1 with 𝑣4.9.201-tegra kernel. They have CUDA 𝑣10.2 with Jetpack 𝑣4.5.1. We use PyTorch 𝑣1.8 and Torchvision 𝑣0.9 as
the DNN training framework. However, Orin requires a more recent OS and library version: CUDA
𝑣11.2, Jetpack 𝑣5.0.1 running on L4T 𝑣34.1.1, Pytorch 𝑣1.12 and Torchvision 𝑣0.13.


