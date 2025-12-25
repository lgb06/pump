import torch
import torch.nn as nn

class TimeSeriesPerturbation(nn.Module):
    def __init__(self, scale_factor=2, truncate_ratio=0.3, 
                 noise_snr_range=5, shift_range=50, speed_scale_factor=2,
                 amplitude_mod_factor=0.2, fault_amp_ratio=0.5, fault_freq_ratio=1.0):
        super(TimeSeriesPerturbation, self).__init__()
        self.scaling_range = (1/scale_factor, scale_factor)
        self.speed_scale_range = (1/speed_scale_factor, speed_scale_factor)
        self.truncate_ratio = truncate_ratio
        self.noise_snr_range = (noise_snr_range, noise_snr_range + 3)
        self.shift_range = shift_range
        self.amplitude_mod_factor = amplitude_mod_factor
        self.fault_amp_ratio = fault_amp_ratio
        self.fault_freq_ratio = fault_freq_ratio
        self.cached_freqs = {}  # Cache for frequency arrays

    def compute_signal_features(self, signal):
        B, L, M = signal.shape
        device = signal.device
        
        # Calculate RMS more efficiently
        rms = torch.sqrt(torch.mean(signal * signal, dim=1, keepdim=True))
        
        # Cache frequency arrays for reuse
        if (L, device) not in self.cached_freqs:
            self.cached_freqs[(L, device)] = torch.fft.rfftfreq(L, device=device)
        freqs = self.cached_freqs[(L, device)]
        
        # Reuse FFT computation if needed
        fft_magnitude = torch.abs(torch.fft.rfft(signal, dim=1))
        arg_max_indices = torch.argmax(fft_magnitude, dim=1, keepdim=True)
        main_freq = freqs[arg_max_indices]
        
        return rms, main_freq

    def time_shift(self, signal):
        shift = torch.randint(-self.shift_range, self.shift_range, (1,), device=signal.device).item()
        # In-place operation
        return torch.roll(signal, shifts=shift, dims=1)

    def speed_scale(self, signal):
        B, L, M = signal.shape
        device = signal.device
        min_speed, max_speed = self.speed_scale_range
        
        # 生成单一缩放因子以减少计算
        scale_factor = torch.tensor([0.0]).uniform_(min_speed, max_speed).to(device).item()
        new_L = int(L * scale_factor)
        
        # 避免中间分配
        if new_L != L:
            # 首先将信号重塑为插值函数所需的格式 [B, C, L]
            reshaped_signal = signal.transpose(1, 2)  # 变为 [B, M, L]
            
            # 应用插值
            reshaped_signal = torch.nn.functional.interpolate(
                reshaped_signal, size=new_L, mode='linear', align_corners=False)
            
            # 再次插值回原始长度
            reshaped_signal = torch.nn.functional.interpolate(
                reshaped_signal, size=L, mode='linear', align_corners=False)
            
            # 转换回原始形状 [B, L, M]
            signal = reshaped_signal.transpose(1, 2)
        
        return signal

    def amplitude_modulation(self, signal):
        B, L, M = signal.shape
        device = signal.device
        rms, main_freq = self.compute_signal_features(signal)
        mod_freq = main_freq * self.fault_freq_ratio
        mod_amp = self.amplitude_mod_factor * rms
        
        # Reuse time index tensor
        if not hasattr(self, 'time_indices') or self.time_indices.shape[1] != L or self.time_indices.device != device:
            self.time_indices = torch.arange(L, device=device).float().view(1, -1, 1)
        
        # Calculate modulation directly
        signal = signal * (1 + mod_amp * torch.cos(2 * torch.pi * mod_freq * self.time_indices / L))
        return signal

    def fault_impact(self, signal):
        B, L, M = signal.shape
        device = signal.device
        rms, main_freq = self.compute_signal_features(signal)
        fault_amp = self.fault_amp_ratio * rms
        fault_freq = main_freq * self.fault_freq_ratio
        
        # Reuse time index tensor
        if not hasattr(self, 'time_indices') or self.time_indices.shape[1] != L or self.time_indices.device != device:
            self.time_indices = torch.arange(L, device=device).float().view(1, -1, 1)
        
        # Add impact directly to signal
        signal = signal + fault_amp * torch.exp(-0.01 * self.time_indices) * torch.cos(2 * torch.pi * fault_freq * self.time_indices / L)
        return signal

    def scale_values(self, signal):
        # Use a single scale factor for the batch
        min_scale, max_scale = self.scaling_range
        #生成一个均值为1的正太分布比例因子，然后将其限制在指定范围内    
        scale_factor =  torch.randn(1).uniform_(min_scale, max_scale).to(signal.device).item()
        signal = signal * scale_factor  # In-place multiplication
        return signal

    def add_gaussian_noise(self, signal):
        # Use a single SNR for the batch to reduce memory usage
        min_snr, max_snr = self.noise_snr_range
        snr_db = torch.tensor([0.0]).uniform_(min_snr, max_snr).to(signal.device).item()
        
        signal_power = torch.mean(signal * signal, dim=1, keepdim=True)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate noise directly into signal
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        signal = signal + noise
        return signal

    def truncate(self, signal):
        B, L, M = signal.shape
        device = signal.device
        
        # Perform FFT
        freq_signal = torch.fft.rfft(signal, dim=1)
        freq_len = freq_signal.shape[1]
        
        # Create mask once for all channels
        mask = torch.rand(1, freq_len, 1, device=device) > self.truncate_ratio
        mask = mask.expand(B, -1, M)
        
        # Apply mask and inverse FFT
        freq_signal = freq_signal * mask.float()
        signal = torch.fft.irfft(freq_signal, n=L, dim=1)
        return signal

    def forward(self, signal, mode=None):
        with torch.no_grad():
            # 创建原始信号的副本用于两种扰动
            signal_condition = signal.clone()
            signal_device = signal.clone()
            
            # 应用工况扰动
            signal_condition = self.time_shift(signal_condition)
            signal_condition = self.speed_scale(signal_condition)
            
            # 应用设备扰动
            signal_device = self.truncate(signal_device)
            signal_device = self.scale_values(signal_device)
            signal_device = self.add_gaussian_noise(signal_device)
            
            # 生成随机权重系数 (0.2-0.8之间的均匀分布)
            alpha = torch.FloatTensor(1).uniform_(0.2, 0.8).item()
            
            # 加权平均两种扰动后的信号
            signal = alpha * signal_condition + (1 - alpha) * signal_device
            
        return signal
if __name__ == "__main__":
    import time
    import numpy as np
    
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建一个随机输入张量，形状为 [128, 2048, 9]
    batch_size = 128
    seq_len = 2048
    channels = 9
    
    print(f"\n创建输入张量: [{batch_size}, {seq_len}, {channels}]")
    input_tensor = torch.randn(batch_size, seq_len, channels, device=device)
    print(f"输入张量形状: {input_tensor.shape}, 设备: {input_tensor.device}")
    
    # 初始化 TimeSeriesPerturbation
    perturbation = TimeSeriesPerturbation(
        scale_factor=2, 
        truncate_ratio=0.3,
        noise_snr_range=5, 
        shift_range=50, 
        speed_scale_factor=2,
        amplitude_mod_factor=0.2, 
        fault_amp_ratio=0.5, 
        fault_freq_ratio=1.0
    ).to(device)
    
    # 测试每一个单独的扰动函数
    distortion_methods = [
        ("time_shift", perturbation.time_shift),
        ("speed_scale", perturbation.speed_scale),
        ("amplitude_modulation", perturbation.amplitude_modulation),
        ("fault_impact", perturbation.fault_impact),
        ("scale_values", perturbation.scale_values),
        ("add_gaussian_noise", perturbation.add_gaussian_noise),
        ("truncate", perturbation.truncate)
    ]
    
    print("\n测试各个扰动函数:")
    for name, method in distortion_methods:
        print(f"\n测试 {name}...")
        # 记录开始时间
        start_time = time.time()
        
        # 记录初始GPU内存
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # 应用扰动
        output = method(input_tensor.clone())
        
        # 同步以确保操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_diff = mem_after - mem_before
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        print(f"  输入形状: {input_tensor.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  处理时间: {elapsed_time:.4f} 秒")
        
        if torch.cuda.is_available():
            print(f"  GPU内存使用: {mem_diff:.2f} MB")
        
        # 检查输入输出是否相同
        is_same_shape = input_tensor.shape == output.shape
        print(f"  形状是否相同: {is_same_shape}")
        
        # 检查数值差异
        if is_same_shape:
            diff = torch.abs(input_tensor - output).mean().item()
            print(f"  平均绝对差异: {diff:.6f}")
    
    # 测试 forward 方法的各个模式
    print("\n测试 forward 方法的各个模式:")
    for mode in ["condition", "fault", "device", None]:
        print(f"\n测试模式: {mode}")
        # 记录开始时间
        start_time = time.time()
        
        # 记录初始GPU内存
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
        
        # 应用扰动
        output = perturbation(input_tensor.clone(), mode=mode)
        
        # 同步以确保操作完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
            mem_diff = mem_after - mem_before
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        print(f"  输入形状: {input_tensor.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  处理时间: {elapsed_time:.4f} 秒")
        
        if torch.cuda.is_available():
            print(f"  GPU内存使用: {mem_diff:.2f} MB")
        
        # 检查输入输出是否相同
        is_same_shape = input_tensor.shape == output.shape
        print(f"  形状是否相同: {is_same_shape}")
    
    print("\n所有测试完成!")