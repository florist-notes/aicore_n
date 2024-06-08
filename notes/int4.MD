# Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training

🌸 [ paper : [web](https://www.usenix.org/conference/nsdi23/presentation/you), [pdf](https://www.usenix.org/system/files/nsdi23-you.pdf) ], [code](https://github.com/ml-energy/zeus) | [ [video](https://youtu.be/aZoD-jgO3fE?si=WhcxloaeZDnFAgvo) ]


<p class="ex1" align="justify" style="padding: 5px 5px 5px 5px">
Abstract:  Training deep neural networks (DNNs) is becoming increasingly more resource- and energy-intensive every year. Unfortunately, existing works primarily focus on optimizing DNN training for faster completion, often without considering the impact on energy efficiency.
In this paper, we observe that common practices to improve training performance can often lead to inefficient energy usage. More importantly, we demonstrate that there is a tradeoff between energy consumption and performance optimization. To this end, we propose Zeus, an optimization framework to navigate this tradeoff by automatically finding optimal job- and GPU-level configurations for recurring DNN training jobs. Zeus uses an online exploration-exploitation approach in conjunction with just-in-time energy profiling, averting the need for expensive offline measurements, while adapting to data drifts over time. Our evaluation shows that Zeus can improve the energy efficiency of DNN training by 15.3%–75.8% for diverse workloads.

</p>

Deep neural networks (DNNs) have seen widespread adoption in various data-driven domains, including computer vision, natural language processing, personalized recommendation, and speech recognition. To support this growth, DNN models are primarily trained on clusters of highly parallel and powerful GPUs . However, the increasing computational demand has led to greater energy consumption. For example, training the GPT-3 model requires 1,287 megawatt-hours (MWh) [[source](https://arxiv.org/abs/2104.10350)], equivalent to the electricity used by an average U.S. household over 120 years. Despite efforts to reduce operational power footprints, like Meta's 28.5% reduction, the energy demand for AI continues to rise. Existing literature on DNN training largely overlooks energy efficiency [[paper](https://dl.acm.org/doi/10.1145/3381831)].

Our research highlights that common performance optimization practices for DNN training often result in inefficient energy usage. Recent works suggest using large batch sizes to enhance training throughput [[paper](https://arxiv.org/abs/1711.00489), [paper](https://arxiv.org/abs/1706.02677)], but this can lower energy efficiency. We found that setting appropriate batch sizes and GPU power limits can reduce energy consumption by 23.8%–74.7% across different workloads.

Reducing energy consumption, however, involves tradeoffs. We discovered a balance between energy use and training time for achieving target accuracy. Our analysis of the energy-time Pareto frontier reveals two key insights: first, all Pareto-optimal configurations offer varying degrees of energy savings compared to using maximum batch sizes and GPU power limits indiscriminately. Second, the reduction in energy consumption often exhibits a non-linear relationship with the increase in training time. This raises the question: how can we automatically identify and manage the tradeoff between energy consumption and training time for DNN training?

In this paper, the author present Zeus, a plug-in optimization framework designed to minimize both energy consumption and training time for DNN training jobs by automatically configuring batch size and GPU power limits. Unlike recent approaches that only consider GPU-specific configurations [[paper](https://www.semanticscholar.org/paper/DUB%3A-Dynamic-Underclocking-and-Bypassing-in-NoCs-Bharadwaj-Das/96e34f96673cca9f118b0bdf5970df5202d4fe84), [paper](https://arxiv.org/pdf/1905.11012)], Zeus addresses both job- and GPU-related configurations without requiring per-job offline profiling or prediction model training [[paper](https://arxiv.org/abs/2201.01684), [paper](https://ieeexplore.ieee.org/document/9139663)]. This makes it particularly suitable for large clusters with heterogeneous hardware and time-varying workloads [[paper](https://www.usenix.org/conference/nsdi22/presentation/weng)].

Zeus employs an online exploration-exploitation approach tailored to DNN training workflows, which need periodic re-training as new data arrives [[paper](https://research.facebook.com/publications/applied-machine-learning-at-facebook-a-datacenter-infrastructure-perspective/) : [pdf](https://systems.cs.columbia.edu/private-systems-class/papers/Hazelwood2018Applied.pdf)]. By automatically exploring various configurations and continuously adjusting based on measured gains or losses, Zeus efficiently navigates the configuration space.

The design of Zeus overcomes two significant challenges: the stochastic nature of DNN training, where energy consumption varies even with identical configurations due to randomness in parameter initialization and data loading [[paper](https://arxiv.org/abs/1806.01427), [paper](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1200)], and the diversity in DNN and GPU architectures, which makes offline profiling results non-transferable [[paper](https://arxiv.org/pdf/1909.06842v6)]. Given the vast configuration space, Zeus decouples batch size and power limit optimization, using a Multi-Armed Bandit (MAB) approach with Thompson Sampling policy for batch size optimization and a just-in-time (JIT) energy profiler for power limit adjustment.

Implemented within the PyTorch framework, Zeus has been evaluated across various workloads, including speech recognition, image classification, NLP, and recommendation tasks. The results show that Zeus reduces energy consumption by 15.3%–75.8% and training time by 60.6% compared to using maximum batch size and GPU power limit settings. Additionally, Zeus quickly converges to the optimal configuration and adapts effectively to data drift, extending its benefits to multi-GPU settings as well.

In summary, the paper's contributions are:

+ Characterizing the energy consumption vs. performance tradeoff for DNN training in terms of job- and GPU-specific configuration parameters.
+ Developing an online optimization framework that adapts to workload dynamics over time.
+ Implementing and evaluating Zeus, which integrates with existing DNN training workflows with minimal code changes and negligible overhead, offering significant benefits.
+ Zeus is open-source and available on [GitHub](https://github.com/ml-energy/zeus).

### `Motivation` : 
In this section, we provide an overview of the energy consumption characteristics of DNN training on GPUs, explore opportunities for reducing energy consumption, and characterize the tradeoff between reducing energy consumption and improving training performance.

### `DNN Training` : 
Modern DNNs are trained by iterating over a large dataset multiple times, with each complete pass called an epoch. One epoch involves thousands of gradient descent iterations over equally sized mini-batches. The batch size affects model accuracy, training throughput, and energy consumption. DNN training performance is typically measured by time-to-accuracy (TTA) for a given target accuracy, where increasing training throughput (or goodput) reduces TTA.

DNNs are predominantly trained on increasingly powerful GPUs, which consume significant energy. Recent benchmarks indicate that GPUs account for approximately 70% of the total energy consumption during DNN training [[paper](https://arxiv.org/abs/2206.05229), [paper](https://ieeexplore.ieee.org/document/9005632)]. In production GPU clusters, as new data flows into the machine learning pipeline, DNNs need periodic re-training, sometimes as frequently as every hour, resulting in recurring jobs in the GPU cluster.

### + Opportunities for Improving Energy Efficiency
We identify two key job and hardware configurations that significantly impact energy efficiency in DNN training: (1) batch size and (2) GPU power limit.

+ `Impact of Batch Size on Energy Efficiency` : The size of each mini-batch during DNN training determines how many samples are processed in one iteration. Larger batch sizes enable faster processing of the entire dataset. However, we observe that common choices of batch size can lead to higher energy consumption for achieving the same target accuracy. We conducted a sweep over a wide range of valid batch sizes (from 8 to the maximum that fits in GPU memory) for six deep learning workloads, including computer vision (CV), natural language processing (NLP), recommendation, and speech recognition on an NVIDIA V100 GPU.
  
<img src="./img/a1.png" width=38%> <img src="./img/a2.png" width=57%>

Our findings indicate that the energy-optimal batch size (Batch Size Opt.) can reduce energy consumption by 3.4%–65.0% compared to the default batch size for the same target accuracy.

## Impact of GPU Power Limit on Energy Efficiency
Setting a GPU's power limit triggers dynamic voltage and frequency scaling (DVFS) to ensure that the power draw does not exceed the set limit [[paper](https://www.sciencedirect.com/science/article/pii/S2352864816300736)]. If not manually configured, the power limit defaults to the maximum setting. We conducted a sweep over various GPU power limits for the previously described setup. Our findings reveal that the optimal energy consumption (Power Limit Opt. in Figure 1) often occurs at a lower power limit than the maximum, resulting in energy savings of 3.0%–31.5%.

## Joint Optimization
Figure 1 illustrates that even greater energy savings (23.8%–74.7%) can be achieved by jointly optimizing both batch size and power limit configurations. Similar opportunities for energy reduction were observed across other GPU generations as well.

### Energy-Performance Tradeoffs :
Optimizing DNN training for energy efficiency typically comes with a tradeoff: a potential increase in training time (TTA). We characterize and explore this tradeoff between energy consumption (ETA) and TTA using DeepSpeech2 trained on the LibriSpeech dataset as an example (Figure 2). Similar results were observed for other workloads.

#### + Tradeoff Between ETA and TTA :
We define the energy consumption of DNN training until it reaches its target accuracy as the energy-to-accuracy (ETA):

$$ ETA (b,p) = TTA (b,p) x AvgPower (b,p) $$

where 𝑝 denotes the GPU power limit, 𝑏 the batch size, and AvgPower ( 𝑏 , 𝑝 ) the average power consumption during training with configuration (b,p). Like TTA, ETA captures the end-to-end goal of DNN training.

Figure 2a shows a scatter plot of (TTA, ETA) for batch size and power limit sweep experiments. Each data point represents the (TTA, ETA) for a specific configuration. We focus on the boundary of all feasible (TTA, ETA) pairs, which are bounded by two straight lines indicating average GPU power consumption. When the GPU is under heavy load, the (TTA, ETA) data points tend towards 210W. Under lighter load, power consumption approaches 90W, close to the GPU’s idle power of 70W.

Importantly, there exists a curve along which all (TTA, ETA) pairs achieve Pareto optimality, we identify a Pareto frontier along which all (TTA, ETA) pairs achieve Pareto optimality [[paper](https://link.springer.com/article/10.1007/BF01442131)]. On this curve, improving ETA without sacrificing TTA is not possible, and vice versa.

Examining the Pareto frontier in Figure 2b, with configurations annotated along each data point, we highlight two key takeaways:

+ Baseline configurations can lead to suboptimal energy efficiency. Furthermore, blindly selecting high batch size and power limit configurations can result in suboptimal TTA.
+ There is a distinct tradeoff between ETA and TTA, with different optimal configurations for each. The configuration optimizing ETA (b=32, p=100W) differs from that optimizing TTA (b=48, p=250W).