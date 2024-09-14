import sys,math

cpu_cores = int(sys.argv[1]) 
no_cpu = math.ceil(cpu_cores / 4)
cpu_frq = int(sys.argv[2])
gpu_frq = int(sys.argv[3])
mem_frq = int(sys.argv[4])

#nvpmodel does not allow setting max value, 0 automatically does this
if mem_frq == 3199000000:
    mem_frq=0

filename = '/etc/nvpmodel.conf'

new_config = [
    '< POWER_MODEL ID=14 NAME=pm_new >\n',
    '\n'.join(f'CPU_ONLINE CORE_{i} 1' if i < cpu_cores else f'CPU_ONLINE CORE_{i} 0' for i in range(12)),
    '\nTPC_POWER_GATING TPC_PG_MASK 0\n',
    'GPU_POWER_CONTROL_ENABLE GPU_PWR_CNTL_EN on\n',
    *[f'CPU_A78_{i} MIN_FREQ {cpu_frq}\nCPU_A78_{i} MAX_FREQ {cpu_frq}\n' for i in range(no_cpu)],
    f'GPU MIN_FREQ {gpu_frq}\n',
    f'GPU MAX_FREQ {gpu_frq}\n',
    f'GPU_POWER_CONTROL_DISABLE GPU_PWR_CNTL_DIS auto\n',
    f'EMC MAX_FREQ {mem_frq}\n',
    'DLA0_CORE MAX_FREQ -1\n',
    'DLA1_CORE MAX_FREQ -1\n',
    'DLA0_FALCON MAX_FREQ -1\n',
    'DLA1_FALCON MAX_FREQ -1\n',
    'PVA0_VPS MAX_FREQ -1\n',
    'PVA0_AXI MAX_FREQ -1\n'
]

with open(filename, 'r') as file:
    lines = file.readlines()

start_idx = None
end_idx = None
last_power_mode_idx = None

for idx, line in enumerate(lines):
    if line.strip().startswith('< POWER_MODEL'):
        last_power_mode_idx = idx
        start_idx = idx
    if last_power_mode_idx is not None and not line.strip():
        end_idx = idx

if start_idx is None or end_idx is None:
    print("Last power mode configuration not found.")

lines[start_idx:end_idx] = new_config

with open(filename, 'w') as file:
    file.writelines(lines)
