# PowerTrain: Fast, generalizable time and power prediction models to optimize DNN training on accelerated edges 🌸

[paper](https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003649?via%3Dihub) | [https://doi.org/10.1016/j.future.2024.07.001](https://doi.org/10.1016/j.future.2024.07.001)  

<p class="ex1" align="justify" style="padding: 5px 5px 5px 5px">
Accelerated edge devices, like Nvidia’s Jetson with 1000+ CUDA cores, are increasingly used for DNN training and federated learning, rather than just for inferencing workloads. A unique feature of these compact devices is their fine-grained control over CPU, GPU, memory frequencies, and active CPU cores, which can limit their power envelope in a constrained setting while throttling the compute performance. Given this vast 10k+ parameter space, selecting a power mode for dynamically arriving training workloads to exploit power–performance trade-offs requires costly profiling for each new workload, or is done ad hoc. We propose PowerTrain, a transfer-learning approach to accurately predict the power and time that will be consumed when we train a given DNN workload (model + dataset) using any specified power mode (CPU/GPU/memory frequencies, core-count). It requires a one-time offline profiling of 1000s of power modes for a reference DNN workload on a single Jetson device (Orin AGX) to build Neural Network (NN) based prediction models for time and power. These NN models are subsequently transferred (retrained) for a new DNN workload, or even a different Jetson device, with minimal additional profiling of just 50 power modes to make accurate time and power predictions. These are then used to rapidly construct the Pareto front and select the optimal power mode for the new workload, e.g., to minimize training time while meeting a power limit. PowerTrain’s predictions are robust to new workloads, exhibiting a low MAPE of <6% for power and <15% for time on six new training workloads (MobileNet, YOLO, BERT, LSTM, etc.) for up to 4400 power modes, when transferred from a ResNet reference workload on Orin AGX. It is also resilient when transferred to two entirely new Jetson devices (Xavier AGX and Jetson Orin Nano) with prediction errors of <14.5% and <11%. These outperform baseline predictions by more than 10% and baseline optimizations by up to 45% on time and 88% on power.

</p>

<img src="./img/powerprofiles.png" width=100%> 

Given the vast parameter space, selecting an optimal power mode to make such trade-offs for training workloads that arrive dynamically is non-trivial. A wrong power mode choice can cause high power and performance penalties, violating time and power load constraints or, in the worst case, destroying the device due to overheating.

It takes 16.3 h to profile even 25% of power modes for Orin AGX for the ResNet model on ImageNet data, and this will not scale for every workload, specially if it arrives dynamically and has time constraints.

                  energy (mWh) = power (mW) × time (h)
