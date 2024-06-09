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
In this section, they provide an overview of the energy consumption characteristics of DNN training on GPUs, explore opportunities for reducing energy consumption, and characterize the tradeoff between reducing energy consumption and improving training performance.

### `DNN Training` : 
Modern DNNs are trained by iterating over a large dataset multiple times, with each complete pass called an epoch. One epoch involves thousands of gradient descent iterations over equally sized mini-batches. The batch size affects model accuracy, training throughput, and energy consumption. DNN training performance is typically measured by time-to-accuracy (TTA) for a given target accuracy, where increasing training throughput (or goodput) reduces TTA.

DNNs are predominantly trained on increasingly powerful GPUs, which consume significant energy. Recent benchmarks indicate that GPUs account for approximately 70% of the total energy consumption during DNN training [[paper](https://arxiv.org/abs/2206.05229), [paper](https://ieeexplore.ieee.org/document/9005632)]. In production GPU clusters, as new data flows into the machine learning pipeline, DNNs need periodic re-training, sometimes as frequently as every hour, resulting in recurring jobs in the GPU cluster.

### + Opportunities for Improving Energy Efficiency
The authors identify two key job and hardware configurations that significantly impact energy efficiency in DNN training: (1) batch size and (2) GPU power limit.

+ `Impact of Batch Size on Energy Efficiency` : The size of each mini-batch during DNN training determines how many samples are processed in one iteration. Larger batch sizes enable faster processing of the entire dataset. However, we observe that common choices of batch size can lead to higher energy consumption for achieving the same target accuracy. We conducted a sweep over a wide range of valid batch sizes (from 8 to the maximum that fits in GPU memory) for six deep learning workloads, including computer vision (CV), natural language processing (NLP), recommendation, and speech recognition on an NVIDIA V100 GPU.
  
<img src="./img/a1.png" width=38%> <img src="./img/a2.png" width=57%>

Author's findings indicate that the energy-optimal batch size (Batch Size Opt.) can reduce energy consumption by 3.4%–65.0% compared to the default batch size for the same target accuracy.

## Impact of GPU Power Limit on Energy Efficiency
Setting a GPU's power limit triggers dynamic voltage and frequency scaling (DVFS) to ensure that the power draw does not exceed the set limit [[paper](https://www.sciencedirect.com/science/article/pii/S2352864816300736)]. If not manually configured, the power limit defaults to the maximum setting. We conducted a sweep over various GPU power limits for the previously described setup. Our findings reveal that the optimal energy consumption (Power Limit Opt. in Figure 1) often occurs at a lower power limit than the maximum, resulting in energy savings of 3.0%–31.5%.

## Joint Optimization
Figure 1 illustrates that even greater energy savings (23.8%–74.7%) can be achieved by jointly optimizing both batch size and power limit configurations. Similar opportunities for energy reduction were observed across other GPU generations as well.

### Energy-Performance Tradeoffs :
Optimizing DNN training for energy efficiency typically comes with a tradeoff: a potential increase in training time (TTA). The authors characterize and explore this tradeoff between energy consumption (ETA) and TTA using DeepSpeech2 trained on the LibriSpeech dataset as an example (Figure 2). Similar results were observed for other workloads.

#### + Tradeoff Between ETA and TTA :
The author defines the energy consumption of DNN training until it reaches its target accuracy as the energy-to-accuracy (ETA):

$$ ETA (b,p) = TTA (b,p) . AvgPower (b,p) $$

where 𝑝 denotes the GPU power limit, 𝑏 the batch size, and AvgPower ( 𝑏 , 𝑝 ) the average power consumption during training with configuration (b,p). Like TTA, ETA captures the end-to-end goal of DNN training.

Figure 2a shows a scatter plot of (TTA, ETA) for batch size and power limit sweep experiments. Each data point represents the (TTA, ETA) for a specific configuration. We focus on the boundary of all feasible (TTA, ETA) pairs, which are bounded by two straight lines indicating average GPU power consumption. When the GPU is under heavy load, the (TTA, ETA) data points tend towards 210W. Under lighter load, power consumption approaches 90W, close to the GPU’s idle power of 70W.

Importantly, there exists a curve along which all (TTA, ETA) pairs achieve Pareto optimality, we identify a Pareto frontier along which all (TTA, ETA) pairs achieve Pareto optimality [[paper](https://link.springer.com/article/10.1007/BF01442131)]. On this curve, improving ETA without sacrificing TTA is not possible, and vice versa.

Examining the Pareto frontier in Figure 2b, with configurations annotated along each data point, we highlight two key takeaways:

+ Baseline configurations can lead to suboptimal energy efficiency. Furthermore, blindly selecting high batch size and power limit configurations can result in suboptimal TTA.
+ There is a distinct tradeoff between ETA and TTA, with different optimal configurations for each. The configuration optimizing ETA (b=32, p=100W) differs from that optimizing TTA (b=48, p=250W).

<img src="./img/zeus.png" width=41%> <img src="./img/zalgo1.png" width=56%>

## Zeus Overview
Zeus is an optimization framework designed to navigate the tradeoff between energy-to-accuracy (ETA) and time-to-accuracy (TTA) by automatically configuring the batch size and GPU power limit for recurring DNN training jobs. It enables developers to optimize energy and/or performance metrics using a single parameter.

### Optimization Metric :
A critical aspect of designing Zeus is defining a cost metric that allows users to express their preference in the ETA-TTA tradeoff. We propose a simple cost metric:

$$ C(b,p;\eta)=\eta⋅ETA(b,p)+(1−\eta)⋅MAXPOWER⋅TTA(b,p)  $$

Here, η is a user-specified parameter indicating the relative importance of energy efficiency versus training performance (throughput). When η=0, the focus is solely on optimizing for time consumption, while η=1 prioritizes energy consumption. MAXPOWER is the maximum power limit supported by the GPU, introduced to unify the units of measure in the cost metric.

## Challenges in Picking the Optimal Configuration
Combining Equations 1 and 2, we have:

$$ C=(\eta⋅AvgPower(b,p)+(1−\eta)⋅MAXPOWER)⋅TTA(b,p) $$

Selecting the optimal configuration(s) to minimize the energy-time cost C for DNN training is challenging due to the large search space [b×p] and the difficulty in efficiently determining the values of both AvgPower(b,p) and TTA(b,p). This complexity arises from the following factors:

+ `Complex Power Consumption Model`: The total energy consumption of a GPU is non-linearly influenced by workload characteristics such as the number of instructions and memory accesses, as well as GPU hardware configurations including core and memory frequency and voltage [[paper](https://dl.acm.org/doi/10.1145/3387902.3392613), [paper](https://dl.acm.org/doi/fullHtml/10.1145/3466752.3480063)]. Existing efforts to estimate GPU energy consumption rely on instruction- or kernel-level information [[paper](https://faculty.cc.gatech.edu/~hyesoon/hong_isca10.pdf), [paper](https://ieeexplore.ieee.org/document/6118939)], which are specific to the architecture and workload.

+ `Stochastic Nature of DNN Training`: Modeling and predicting the duration required to train a specific model to target accuracy (TTA) is inherently difficult [[paper](https://www.usenix.org/conference/nsdi19/presentation/gu)]. Additionally, randomness in model initialization and data loading leads to TTA variations, even when the same job is executed on the same GPU with the same configuration—such variations can be as large as 14% [[paper](https://arxiv.org/abs/1806.01427)].

Fortunately, DNN training jobs often recur in production clusters [[paper](https://ieeexplore.ieee.org/document/8327042), [paper](https://www.usenix.org/conference/nsdi22/presentation/weng)]. This recurrence provides opportunities for empirical estimation through repeated measurements across instances of the same training job.

## Architectural Overview
Zeus employs an online exploration-exploitation approach to minimize the aggregate cost of recurrent DNN training jobs. It addresses the challenges of optimizing energy and performance tradeoffs through two key components:

+ `Just-in-Time (JIT) Online Profiler`: This component efficiently profiles the energy characteristics of the training job in real-time.
+ `Multi-Armed Bandit (MAB) with Thompson Sampling`: This component handles the stochastic nature of deep learning training and optimizes under uncertainty, adapting to changing workloads such as data drift.

The combination of the JIT profiler and MAB makes Zeus a fully online solution, allowing immediate optimization for incoming jobs.

### Workflow of Zeus :
Figure 3 provides an overview of the high-level workflow of Zeus:

+ `Job Submission`: In a production environment, users submit recurrent DNN training jobs to Zeus. Each job is a tuple consisting of data, model, optimizer, and the target validation metric, along with a set of feasible batch sizes B and power limits P to explore.
+ `Configuration Prediction`: Zeus predicts the optimal batch size and power limit configuration based on past execution history.
+ `Job Launch`: The training job is launched with the predicted configuration.
+ `Data Collection and Feedback`: During and after the training process, statistics about DNN training (e.g., validation metric) and GPU power consumption are collected and fed back to the Zeus optimizer. The optimizer learns from the feedback and adjusts its internal states.

The training job will be terminated upon either reaching the target metric or exceeding a stopping threshold determined by Zeus. This automated feedback loop minimizes the key objective of energy-time cost.

Building Zeus requires both algorithm design and systems support. The next sections describe the core optimization algorithm details and the implementation highlights of Zeus.

## Zeus Algorithm Design
This section details how Zeus selects the optimal batch size and GPU power limit to minimize the overall cost of recurrent DNN training tasks. We start with the problem formulation and describe the decoupling of batch size and power limit optimizations. We then explain the optimization of the power limit and batch size within this decoupled framework, concluding with a discussion on addressing common challenging scenarios.

#### Problem Formulation :
Zeus aims to minimize the cost of a recurring job by exploring the feasible set of batch sizes B and power limits P. The goal is to balance the tradeoff between exploration and exploitation to find the optimal configuration without incurring excessive costs. The objective, based on the cost function from Equation 2, is to:

$$ min_{b,p} \sum_{t=1}^{T} C(b_t, p_t; \eta) $$
$$ subject to b_t \epsilon B, p_t \epsilon P, \forall t \epsilon[1,T] $$

Here, b_t and p_t are the batch size and power limit chosen at the t-th recurrence of the job, and b and p are vectors of length T.

The problem is complex due to the vast search space and the requirement to run DNN training to obtain each value of C(b,p;η). However, by expanding the cost function (Equation 3), we can decouple the exploration of batch size and power limit, making the problem more tractable:

$$ C(b,p;\eta)=(\eta⋅AvgPower(b,p)+(1−\eta)⋅MAXPOWER)⋅TTA(b,p) $$
$$ = Epochs(b) \frac{\eta⋅AvgPower(b,p)+(1−\eta)⋅MAXPOWER}{Throughput(b,p)} $$

where Epochs(b) denotes the number of epochs needed to reach the target, and Throughput(b,p) is the number of epochs per second.

Two key insights allow the decoupling of batch size b and power limit p:

Profiling Efficiency: Given b, AvgPower(b,p) and Throughput(b,p) can be quickly profiled during training for all possible choices of p. This is due to the iterative nature of DNN training, yielding stable power and throughput estimations with a small number of iterations.

Independence of Epochs: Epochs(b) is unaffected by the choice of p since changing the power limit does not change what is computed.

Thus, the optimal power limit for any batch size can be determined independently through online profiling. Each choice of batch size is automatically paired with the optimal power limit, reducing the search space to the set of batch sizes B.

Formally, the problem is decoupled into a two-level optimization problem:

$$ min_{b \epsilon B^T} \sum_{t=1}^{T} Epochs(b_t)⋅EpochCost(b_t; \eta) $$

where,

$$ EpochCost(b_t;\eta) = min_{p_t \epsilon P} \frac{\eta⋅AvgPower(b_t,p_t)+(1−\eta)⋅MAXPOWER}{Throughput(b_t,p_t)} $$


When a job arrives, Zeus first decides the batch size to use. Then, based on the selected batch size, Zeus determines the optimal power limit.

### Optimizing Power Limit
To optimize the power limit for a given batch size b:

$$ EpochCost(b;\eta) = min_{p \epsilon P} \frac{\eta⋅AvgPower(b,p)+(1−\eta)⋅MAXPOWER}{Throughput(b,p)} $$

Zeus profiles the power consumption and throughput for all possible power limits during the initial iterations of the training process, enabling the selection of the optimal power limit that minimizes the cost for the given batch size.

### Optimizing Batch Size
To optimize the batch size:

$$ min_{b \epsilon B^T} \sum_{t=1}^{T} Epochs(b_t)⋅EpochCost(b_t; \eta) $$

Zeus utilizes a Multi-Armed Bandit (MAB) approach with Thompson Sampling to explore and exploit batch sizes, adapting to the stochastic nature of DNN training and varying workloads. The MAB framework helps balance the tradeoff between exploring new configurations and exploiting known good configurations.

<img src="./img/zalgo2.png" width=54%><img src="./img/zeus_gauss.png" width=44%>
<img src="./img/zeus2.png" width=100%>

### Addressing Common Challenges
Zeus addresses several common challenges in optimizing DNN training configurations:

+ `Handling Stochasticity`: By using Thompson Sampling in the MAB framework, Zeus adapts to the inherent randomness in DNN training, such as variations in TTA due to model initialization and data loading.
+ `Dynamic Workloads`: Zeus continuously profiles and adapts to changing workloads, such as data drift, ensuring optimal configurations over time.
+ `Scalability`: The decoupled optimization approach reduces the complexity of the search space, allowing Zeus to scale efficiently with larger clusters and more diverse hardware configurations.

Through these mechanisms, Zeus effectively minimizes the energy-time cost for recurrent DNN training tasks in dynamic production environments.






