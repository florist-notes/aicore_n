# PowerTrain: Fast, generalizable time and power prediction models to optimize DNN training on accelerated edges 🌸

[paper](https://www.sciencedirect.com/science/article/abs/pii/S0167739X24003649?via%3Dihub) | [https://doi.org/10.1016/j.future.2024.07.001](https://doi.org/10.1016/j.future.2024.07.001)  

<p class="ex1" align="justify" style="padding: 5px 5px 5px 5px">
Accelerated edge devices, like Nvidia’s Jetson with 1000+ CUDA cores, are increasingly used for DNN training and federated learning, rather than just for inferencing workloads. A unique feature of these compact devices is their fine-grained control over CPU, GPU, memory frequencies, and active CPU cores, which can limit their power envelope in a constrained setting while throttling the compute performance. Given this vast 10k+ parameter space, selecting a power mode for dynamically arriving training workloads to exploit power–performance trade-offs requires costly profiling for each new workload, or is done ad hoc. We propose PowerTrain, a transfer-learning approach to accurately predict the power and time that will be consumed when we train a given DNN workload (model + dataset) using any specified power mode (CPU/GPU/memory frequencies, core-count). It requires a one-time offline profiling of 1000s of power modes for a reference DNN workload on a single Jetson device (Orin AGX) to build Neural Network (NN) based prediction models for time and power. These NN models are subsequently transferred (retrained) for a new DNN workload, or even a different Jetson device, with minimal additional profiling of just 50 power modes to make accurate time and power predictions. These are then used to rapidly construct the Pareto front and select the optimal power mode for the new workload, e.g., to minimize training time while meeting a power limit. PowerTrain’s predictions are robust to new workloads, exhibiting a low MAPE of <6% for power and <15% for time on six new training workloads (MobileNet, YOLO, BERT, LSTM, etc.) for up to 4400 power modes, when transferred from a ResNet reference workload on Orin AGX. It is also resilient when transferred to two entirely new Jetson devices (Xavier AGX and Jetson Orin Nano) with prediction errors of <14.5% and <11%. These outperform baseline predictions by more than 10% and baseline optimizations by up to 45% on time and 88% on power.

</p>

<img src="./img/powerprofiles.png" width=100%> 

Given the vast parameter space, selecting an optimal power mode to make such trade-offs for training workloads that arrive dynamically is non-trivial. A wrong power mode choice can cause high power and performance penalties, violating time and power load constraints or, in the worst case, destroying the device due to overheating.

It takes 16.3 h to profile even 25% of power modes for Orin AGX for the ResNet model on ImageNet data, and this will not scale for every workload, specially if it arrives dynamically and has time constraints.

                  energy (mWh) = power (mW) × time (h)

PowerTrain :

<img src="./img/powertrain.png" width=100%> 

### Proposed Approach :

The authors propose two data-driven approaches to address the challenge. First, they train two Neural Network (NN) models—one for time predictions and one for power predictions—on a subset of power modes for a given DNN workload. These models make predictions for unseen power modes within the same workload.

Our second approach, PowerTrain, trains NN models on a large corpus of power modes measured from a reference DNN workload. When a new DNN workload arrives, PowerTrain uses transfer learning on telemetry from profiling approximately 50 power modes to retrain the models. These updated models estimate training time and power for all possible power modes, enabling quick Pareto front analysis across time and power to meet optimization goals, such as the fastest training time within a power limit or the lowest power within a time budget.

PowerTrain has lower profiling overheads than NN for new workloads and offers competitive prediction accuracy. It requires a one-time profiling overhead on the reference workload to train the initial NN models, but incremental overhead for new workloads is limited to profiling tens of power modes. PowerTrain generalizes to new DNN workloads, datasets, and even new device types, such as transitioning from Orin AGX to Xavier AGX. In contrast, NN models trained from scratch for new workloads need twice as much profiling data to achieve similar accuracy as PowerTrain.

### Representative Results Against Baselines:

<b>Predictions </b>:
Nvidia offers a PowerEstimator tool (NPE) to estimate the power usage by Orin AGX for a specific power mode. We used NPE and our PowerTrain (PT) approach, with ResNet as the reference workload on Orin AGX, to predict the power usage for three DNN workloads across two diverse power modes each. We reported the Mean Average Percentage Error (MAPE %) relative to the actual observed power usage. Except for PM1 for ResNet, where PT's error (5%) was slightly higher than NPE's (4%), PT provided better predictions in all other cases while NPE consistently overestimated.

<b>Optimization </b>:
Having predictions for power and training time for a DNN workload for any given power mode helps users optimize the system to meet different goals. Nvidia recommends three default power modes: 15W, 30W, and 50W. We used the best of these three to meet the optimization goal and compared it to our PowerTrain models, which discovered a Pareto front to select a custom power mode. PowerTrain (PT) had the fewest solutions exceeding the optimal for 5 out of 6 cases (except ResNet 15W) compared to Nvidia’s suggestions (NV), while generally remaining within power limits.

<img src="./img/ptcmp.png" width=100%> 

PT-based optimization also outperformed simpler baselines like choosing the default MAXN power mode and random profiling (RND) of 50 power modes. For optimization problems with power limits from 17W to 50W, PT had the lowest time penalty of 1% above the optimal while also having the lowest percentage (26.5%) of solutions exceeding the power limit by over 1W. This low time penalty is beneficial for long training runs over several epochs. For instance, training YOLO takes 200 epochs (~49 hours) on the Orin, and PT's 12% time benefit reduces this by 5.88 hours. Similarly, MobileNet's 148 epochs (~50 hours) see a 6.5-hour reduction when optimized using PT.

#### Discussion :

For instance, a one-time training workload on an edge service that runs a new and large DNN model on a vast corpus of user data may benefit from a brute force approach. Even if profiling time is costly, the perfect power mode is crucial for time-consuming tasks, which could span days. However, this is impractical for edge workloads.

More common on the edge is fine-tuning a pre-trained DNN on a smaller dataset, such as model personalization on a private edge device, where training typically lasts a few hours. Here, using our NN approach with profiling over hundreds of power modes is feasible due to the longer training time.

In continuous learning scenarios, where the same DNN is retrained regularly with small batches of new data to address data drift, PowerTrain is ideal due to its smaller data collection time. For edge devices in a federated learning setup or a private edge-cloud supporting multiple applications, with frequent and unknown duration DNN training workloads, PowerTrain minimizes data collection time and meets optimization criteria before workload changes.


### Contributions :
The authors make the following contributions:

+ `Neural Network (NN) Prediction Approach`: Developed an NN-based approach to estimate time and power for DNN training workloads on the Nvidia Jetson Orin AGX for any given power mode using a large profiling corpus. Extended this into PowerTrain (PT), a transfer learning-based approach that significantly reduces online profiling overheads for new DNN workloads on Orin or other Jetson devices.

+ `Validation Across Multiple Workloads and Devices`: Validated NN and PT for accuracy and generalizability across 7 different DNN workloads and three Jetson edge devices (Orin AGX, Xavier AGX, and Orin Nano), as well as across 3 different training minibatch sizes on 2 workloads.

+ `Optimization of Training Time and Power Limits`: Applied PT predictions to optimize training time and power limit trade-offs for several DNN workloads, demonstrating superior performance compared to other baselines, including NN.