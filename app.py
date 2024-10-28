import threading
import queue
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from dataclasses import dataclass
from PIL import Image, ImageTk
import pickle
import logging
import os
import sys
import scipy.signal as signal  # Importing signal to fix the 'signal' not defined error

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fractal_being.log")
    ]
)

@dataclass
class BeingConfig:
    # Video settings
    video_width: int = 160
    video_height: int = 120
    fps: int = 15
    
    # Audio settings
    sample_rate: int = 44100
    audio_window: int = 1024
    audio_hop: int = 512
    
    # Neural settings
    hidden_dim: int = 128
    max_neurons: int = 1000
    max_depth: int = 4
    
    # System settings
    feedback_strength: float = 0.1
    update_rate: int = 30  # Hz
    
    # Feature dimensions
    def calculate_feature_dim(self):
        # 32 for audio + DCT features based on resolution
        dct_size = min(64, (self.video_width * self.video_height) // (16 * 16))
        return 32 + dct_size  # audio features + adaptive video features
    
    @property
    def feature_dim(self):
        return self.calculate_feature_dim()

class FractalDimensionAdapter(nn.Module):
    """Advanced dimension adaptation for fractal neural networks with adaptive scaling"""
    def __init__(self, input_dim: int, output_dim: int, adaptation_levels: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adaptation_levels = adaptation_levels
        
        # Adaptive input handling
        self.input_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        
        # Multi-scale processing paths
        self.scale_paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * (2**i), input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            ) for i in range(self.adaptation_levels)
        ])
        
        # Attention mechanism for scale weighting
        self.scale_attention = nn.Sequential(
            nn.Linear(output_dim * adaptation_levels, adaptation_levels),
            nn.Softmax(dim=-1)
        )
        
        # Dynamic feature selection
        self.feature_gates = nn.Parameter(torch.ones(output_dim))
        self.feature_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Dimension reconciliation with residual connection
        self.reconciliation = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Adaptive pooling for flexible input handling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(input_dim)
        
        # State tracking
        self.register_buffer('running_mean', torch.zeros(output_dim))
        self.register_buffer('running_var', torch.ones(output_dim))
        self.register_buffer('scale_weights_history', torch.zeros(100, adaptation_levels))
        self.history_pos = 0
        
    def _ensure_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is a proper tensor with correct dtype"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dtype != torch.float32:
            x = x.float()
        return x
    
    def _create_multiscale_input(self, x: torch.Tensor) -> list:
        """Create multi-scale versions of input with proper dimension handling"""
        try:
            scales = []
            
            # Base scale
            x = self._ensure_tensor(x)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            # Handle different input sizes
            if x.shape[-1] != self.input_dim:
                x = x.unsqueeze(1)  # Add channel dim
                x = self.adaptive_pool(x)
                x = x.squeeze(1)
            
            scales.append(x)
            
            # Create additional scales
            for i in range(1, self.adaptation_levels):
                # Group features for larger scale
                scale_size = self.input_dim * (2**i)
                if x.shape[-1] >= scale_size:
                    # Use actual data for larger scales
                    scaled = x.reshape(x.shape[0], -1, scale_size).mean(dim=1)
                else:
                    # Pad for required scale size
                    pad_size = scale_size - x.shape[-1]
                    scaled = F.pad(x, (0, pad_size), mode='replicate')
                scales.append(scaled)
            
            return scales
                
        except Exception as e:
            logging.error(f"Error in multi-scale creation: {e}")
            return [torch.zeros((1, self.input_dim)) for _ in range(self.adaptation_levels)]
    
    def _apply_feature_gating(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic feature gating with learned importance"""
        gates = torch.sigmoid(self.feature_gates)
        return x * gates + self.feature_bias
    
    def _update_running_stats(self, x: torch.Tensor):
        """Update running statistics for normalization"""
        with torch.no_grad():
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            momentum = 0.1
            self.running_mean = (1 - momentum) * self.running_mean + momentum * mean
            self.running_var = (1 - momentum) * self.running_var + momentum * var
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Create multi-scale versions
            scales = self._create_multiscale_input(x)
            
            # Process each scale
            scale_outputs = []
            for scale, path in zip(scales, self.scale_paths):
                processed = path(scale)
                scale_outputs.append(processed)
            
            # Stack scale outputs
            combined = torch.cat(scale_outputs, dim=-1)
            
            # Calculate attention weights for scales
            scale_weights = self.scale_attention(combined)
            
            # Update scale weights history
            self.scale_weights_history[self.history_pos] = scale_weights.mean(dim=0)
            self.history_pos = (self.history_pos + 1) % self.scale_weights_history.shape[0]
            
            # Apply attention weights
            scale_outputs = torch.stack(scale_outputs, dim=-1)
            attended = torch.sum(scale_outputs * scale_weights.unsqueeze(1), dim=-1)
            
            # Apply feature gating
            gated = self._apply_feature_gating(attended)
            
            # Final reconciliation with residual connection
            output = self.reconciliation(gated) + gated
            
            # Update running statistics
            self._update_running_stats(output)
            
            return output
                
        except Exception as e:
            logging.error(f"Error in FractalDimensionAdapter forward: {e}")
            return torch.zeros((1, self.output_dim))
    
    def get_adaptation_stats(self) -> dict:
        """Get statistics about the adaptation process"""
        try:
            return {
                'scale_importance': self.scale_weights_history.mean(dim=0).tolist(),
                'feature_gates': torch.sigmoid(self.feature_gates).tolist(),
                'running_mean': self.running_mean.tolist(),
                'running_var': self.running_var.tolist()
            }
        except Exception as e:
            logging.error(f"Error getting adaptation stats: {e}")
            return {}
    
    def adapt_to_distribution(self, x: torch.Tensor):
        """Adapt to input distribution changes"""
        try:
            # Update feature gates based on input statistics
            with torch.no_grad():
                mean = x.mean(dim=0)
                var = x.var(dim=0)
                
                # Adjust gates based on feature importance
                importance = var / (mean.abs() + 1e-6)
                self.feature_gates.data = 0.9 * self.feature_gates + 0.1 * importance
                
                # Adjust bias based on mean
                self.feature_bias.data = 0.9 * self.feature_bias + 0.1 * (-mean)
                
        except Exception as e:
            logging.error(f"Error in distribution adaptation: {e}")

class SpectralProcessor:
    def __init__(self, config: BeingConfig):
        self.config = config
        self.mel_filters = self._create_mel_filterbank(
            num_filters=32,
            fft_size=self.config.audio_window,
            sample_rate=self.config.sample_rate
        )
        
    def _create_mel_filterbank(self, num_filters, fft_size, sample_rate):
        mel_max = 2595 * np.log10(1 + (sample_rate/2)/700)
        mel_points = np.linspace(0, mel_max, num_filters + 2)
        hz_points = 700 * (10**(mel_points/2595) - 1)
        
        bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
        
        fbank = np.zeros((num_filters, fft_size//2 + 1))
        for i in range(num_filters):
            for j in range(bins[i], bins[i+1]):
                fbank[i, j] = (j - bins[i]) / (bins[i+1] - bins[i])
            for j in range(bins[i+1], bins[i+2]):
                fbank[i, j] = (bins[i+2] - j) / (bins[i+2] - bins[i+1])
                
        return torch.FloatTensor(fbank)
    
    def audio_to_features(self, audio: np.ndarray) -> torch.Tensor:
        try:
            _, _, Zxx = signal.stft(
                audio,
                fs=self.config.sample_rate,
                nperseg=self.config.audio_window,
                noverlap=self.config.audio_window - self.config.audio_hop
            )
            
            mag = np.abs(Zxx)
            mel_features = torch.matmul(self.mel_filters, torch.FloatTensor(mag))
            mel_features = torch.log(mel_features + 1e-6)
            mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-6)
            
            return mel_features.flatten()
        except Exception as e:
            logging.error(f"Error in audio_to_features: {e}")
            return torch.zeros(32)
    
    def video_to_features(self, frame: np.ndarray) -> torch.Tensor:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Adaptive resize based on resolution
            target_size = int(np.sqrt(self.config.feature_dim - 32))  # Subtract audio features
            if target_size < 1:
                target_size = 1
            resized = cv2.resize(gray, (target_size, target_size))
            dct = cv2.dct(resized.astype(np.float32))
            
            features = []
            for i in range(target_size):
                for j in range(target_size - i):
                    features.append(dct[i][j])
                    if i != j:
                        features.append(dct[j][i])
                        
            # Take appropriate number of features
            features = torch.FloatTensor(features[:self.config.feature_dim - 32])
            if features.numel() == 0:
                features = torch.zeros(self.config.feature_dim - 32)
            else:
                features = (features - features.mean()) / (features.std() + 1e-6)
            
            return features
        except Exception as e:
            logging.error(f"Error in video_to_features: {e}")
            return torch.zeros(self.config.feature_dim - 32)

class FractalNeuron(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, depth: int = 0, max_depth: int = 3):
        super().__init__()
        self.depth = depth
        self.max_depth = max_depth
        
        # Dimension adapter
        self.dimension_adapter = FractalDimensionAdapter(input_dim, input_dim)
        
        # Core processing
        self.core = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )
        
        # Recursive children
        self.child_neurons = nn.ModuleList()
        if depth < max_depth:
            child_count = max(1, 2 - depth)
            for _ in range(child_count):
                child = FractalNeuron(
                    input_dim=output_dim,
                    output_dim=output_dim,
                    depth=depth + 1,
                    max_depth=max_depth
                )
                self.child_neurons.append(child)
        
        # State tracking
        self.activation_state = 0.0
        self.energy = 100.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Adapt dimensions
            x = self.dimension_adapter(x)
            
            # Process through core
            x = self.core(x)
            
            # Update state
            self.activation_state = x.mean().item()
            self.energy = max(0.0, self.energy - 0.1 * (self.depth + 1))
            
            # Process through children
            if self.child_neurons and self.energy > 0:
                child_outputs = []
                for child in self.child_neurons:
                    try:
                        child_out = child(x)
                        child_outputs.append(child_out)
                    except Exception as e:
                        logging.error(f"Error in child processing: {e}")
                        continue
                
                if child_outputs:
                    try:
                        x = torch.stack(child_outputs).mean(dim=0)
                    except Exception as e:
                        logging.error(f"Error combining child outputs: {e}")
            
            return x
                
        except Exception as e:
            logging.error(f"Error in FractalNeuron forward: {e}")
            return torch.zeros((1, self.core[0].out_features))

class FractalBrain(nn.Module):
    def __init__(self, config: BeingConfig):
        super().__init__()
        self.config = config
        
        # Store dimensions
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        
        # Feature compression
        self.feature_compress = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh()
        )
        
        # Neural pathways with consistent dimensions
        self.sensory = FractalNeuron(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            max_depth=config.max_depth
        )
        
        self.processor = FractalNeuron(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            max_depth=config.max_depth
        )
        
        self.generator = FractalNeuron(
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            max_depth=config.max_depth
        )
        
        # Feature expansion
        self.feature_expand = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh()
        )
        
        # Memory buffer
        self.memory = torch.zeros(100, self.hidden_dim)
        self.memory_pos = 0

    def _ensure_tensor(self, x) -> torch.Tensor:
        """Ensure input is a proper tensor"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dtype != torch.float32:
            x = x.float()
        return x

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input to correct dimensions with proper error handling"""
        try:
            # Ensure proper tensor
            x = self._ensure_tensor(x)
            
            # Get total elements
            total_elements = x.numel()
            
            # Handle empty tensor
            if total_elements == 0:
                return torch.zeros((1, self.feature_dim))
            
            # Flatten if needed
            if len(x.shape) > 2:
                x = x.reshape(-1)
            
            # Handle different input cases
            if total_elements < self.feature_dim:
                # Pad if too small
                x = F.pad(x.reshape(-1), (0, self.feature_dim - total_elements))
                x = x.unsqueeze(0)
            elif total_elements > self.feature_dim:
                # Use adaptive pooling for dimension reduction
                x = x.reshape(1, 1, -1)
                x = F.adaptive_avg_pool1d(x, self.feature_dim)
                x = x.squeeze(1)
            else:
                # Just reshape if exactly right
                x = x.reshape(1, self.feature_dim)
            
            # Final shape check
            if x.shape[-1] != self.feature_dim:
                logging.warning(f"Unexpected shape after reshape: {x.shape}")
                x = F.interpolate(
                    x.unsqueeze(1), 
                    size=self.feature_dim, 
                    mode='linear'
                ).squeeze(1)
            
            return x
                
        except Exception as e:
            logging.error(f"Error in _reshape_input: {e}")
            return torch.zeros((1, self.feature_dim))

    def _process_memory(self, current_state: torch.Tensor):
        """Update and maintain memory buffer"""
        try:
            # Ensure proper shape
            if len(current_state.shape) > 2:
                current_state = current_state.reshape(1, -1)
            
            # Update memory
            self.memory[self.memory_pos] = current_state.detach().mean(dim=0)
            self.memory_pos = (self.memory_pos + 1) % self.memory.shape[0]
            
        except Exception as e:
            logging.error(f"Error in _process_memory: {e}")

    def _validate_size(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has valid size and content"""
        try:
            # Check for zero-size tensor
            if x.numel() == 0:
                return torch.zeros((1, self.feature_dim))
                
            # Check for NaN or Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                return torch.zeros((1, self.feature_dim))
                
            # Ensure minimum size
            if x.numel() < self.feature_dim:
                # Pad to feature_dim
                if len(x.shape) == 1:
                    x = F.pad(x, (0, self.feature_dim - x.numel()))
                else:
                    x = F.pad(x.reshape(-1), (0, self.feature_dim - x.numel())).reshape(1, -1)
                    
            # Ensure maximum size
            elif x.numel() > self.feature_dim:
                # Truncate to feature_dim
                x = x.reshape(-1)[:self.feature_dim]
                
            # Reshape if needed
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
                
            return x.float()
                
        except Exception as e:
            logging.error(f"Error in size validation: {e}")
            return torch.zeros((1, self.feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Validate input size
            x = self._validate_size(x)
            
            # Log shapes for debugging
            logging.debug(f"Input shape after reshape: {x.shape}")
            
            # Compress features
            x = self.feature_compress(x)
            
            # Process through neural pathways
            try:
                sensory_out = self.sensory(x)
                processed = self.processor(sensory_out)
                generated = self.generator(processed)
            except Exception as e:
                logging.error(f"Error in neural processing: {e}")
                return torch.zeros((1, self.feature_dim))
            
            # Update memory with processed state
            self._process_memory(processed)
            
            # Expand back to feature dimension
            output = self.feature_expand(generated)
            
            # Ensure output has correct shape
            output = output.reshape(-1)[:self.feature_dim]
            
            # Final validation
            if torch.isnan(output).any():
                logging.warning("NaN detected in output, returning zeros")
                return torch.zeros((1, self.feature_dim))
            
            return output.unsqueeze(0)
            
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            return torch.zeros((1, self.feature_dim))

    def get_state_dict_custom(self) -> dict:
        """Get complete state for saving"""
        try:
            return {
                'model': self.state_dict(),
                'memory': self.memory.clone(),
                'memory_pos': self.memory_pos,
                'config': {
                    'feature_dim': self.feature_dim,
                    'hidden_dim': self.hidden_dim
                }
            }
        except Exception as e:
            logging.error(f"Error getting state dict: {e}")
            return None

    def load_state_dict_custom(self, state_dict: dict):
        """Load state with compatibility checking"""
        try:
            if 'model' in state_dict:
                # Load model state
                super().load_state_dict(state_dict['model'])
                
                # Load memory if dimensions match
                if 'memory' in state_dict and state_dict['memory'].shape == self.memory.shape:
                    self.memory = torch.from_numpy(state_dict['memory'].numpy())
                    
                # Load memory position
                if 'memory_pos' in state_dict:
                    self.memory_pos = state_dict['memory_pos']
                    
                logging.info("State loaded successfully")
            else:
                logging.warning("Invalid state dict format")
                
        except Exception as e:
            logging.error(f"Error loading state dict: {e}")

class FractalBeing:
    def __init__(self, config: BeingConfig):
        self.config = config
        # Initialize SpectralProcessor and FractalBrain
        self.spectral = SpectralProcessor(config)
        self.brain = FractalBrain(config)
        
        self.setup_av_capture()
        self.setup_av_synthesis()
        
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        self.input_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=30)
        
    def setup_av_capture(self):
        try:
            self.video_capture = cv2.VideoCapture(0)  # Change this if your webcam is not 0 
            if not self.video_capture.isOpened():
                logging.error("Failed to open video capture device.")
                raise ValueError("Could not open webcam. Please check the device index.")
            
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video_height)
            self.video_capture.set(cv2.CAP_PROP_FPS, self.config.fps)
            logging.info("Video capture initialized.")
            
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.audio_window
            )
            logging.info("Audio capture stream initialized.")
        except Exception as e:
            logging.error(f"Error setting up AV capture: {e}")
            raise
        
    def setup_av_synthesis(self):
        try:
            self.audio_output = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=self.config.audio_window
            )
            logging.info("Audio synthesis stream initialized.")
        except Exception as e:
            logging.error(f"Error setting up AV synthesis: {e}")
            raise

    def capture_input(self):
        while self.running:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    logging.warning("Failed to read frame from video capture.")
                    continue
                    
                audio_data = np.frombuffer(
                    self.audio_stream.read(self.config.audio_window, exception_on_overflow=False),
                    dtype=np.float32
                )
                
                if audio_data.size == 0:
                    logging.warning("Captured empty audio data.")
                    continue
                
                video_features = self.spectral.video_to_features(frame)
                audio_features = self.spectral.audio_to_features(audio_data)
                
                if audio_features.numel() == 0 or video_features.numel() == 0:
                    logging.warning("Empty features detected. Skipping enqueue.")
                    continue
                
                combined = torch.cat([audio_features.flatten(), video_features])
                
                if not self.input_queue.full():
                    self.input_queue.put((combined, frame, audio_data))
                    
            except Exception as e:
                logging.error(f"Error in capture_input: {e}")
                time.sleep(0.1)

    def process_loop(self):
        while self.running:
            try:
                logging.debug(f"Input queue size: {self.input_queue.qsize()}")
                logging.debug(f"Output queue size: {self.output_queue.qsize()}")
                
                if self.input_queue.empty():
                    time.sleep(0.01)
                    continue
                    
                features, frame, audio = self.input_queue.get()
                
                # Ensure features tensor has correct shape
                if len(features.shape) == 1:
                    features = features.unsqueeze(0)
                    
                with torch.no_grad():
                    output = self.brain(features)
                
                # Handle output reshaping
                if len(output.shape) == 1:
                    output = output.unsqueeze(0)
                
                # Split features ensuring we have enough elements
                num_audio = 32  # Fixed number for audio features
                if output.shape[-1] < num_audio:
                    logging.warning(f"Output feature size {output.shape[-1]} is less than expected audio features {num_audio}. Padding.")
                    audio_out = torch.cat([output[0, :].flatten(), torch.zeros(num_audio - output.shape[-1])])
                else:
                    audio_out = output[0, :num_audio].cpu().numpy()
                
                # Pad audio if needed
                if len(audio_out) < num_audio:
                    audio_out = np.pad(audio_out, (0, num_audio - len(audio_out)), 'constant')
                
                # Calculate video dimensions
                video_elements = output.shape[-1] - num_audio
                video_side = int(np.sqrt(video_elements))
                
                # Handle case where video_elements is not a perfect square
                if video_side * video_side != video_elements:
                    video_side = max(1, video_side)  # Ensure at least 1
                    
                # Reshape video ensuring square dimensions
                video_out = output[0, num_audio:num_audio + video_side * video_side].cpu().numpy()
                video_out = video_out.reshape(video_side, video_side)
                
                # Process audio
                synth_audio = signal.resample(
                    audio_out,
                    self.config.audio_window
                ).astype(np.float32)
                
                # Process video
                video_frame = cv2.resize(
                    video_out,
                    (self.config.video_width, self.config.video_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Normalize outputs
                synth_audio = np.clip(synth_audio, -1, 1).astype(np.float32)
                video_frame = np.clip(video_frame * 255, 0, 255).astype(np.uint8)
                
                if not self.output_queue.full():
                    self.output_queue.put((video_frame, synth_audio))
                    
                self.audio_output.write(synth_audio.tobytes())
                
            except Exception as e:
                logging.error(f"Error in process_loop: {e}")
                time.sleep(0.1)

    def start(self):
        try:
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_input, daemon=True)
            self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
            self.capture_thread.start()
            self.process_thread.start()
            logging.info("Fractal Being started.")
        except Exception as e:
            logging.error(f"Error starting FractalBeing: {e}")
            self.stop()

    def stop(self):
        try:
            self.running = False
            
            if self.capture_thread:
                self.capture_thread.join(timeout=1)
            if self.process_thread:
                self.process_thread.join(timeout=1)
                
            if hasattr(self, 'video_capture') and self.video_capture.isOpened():
                self.video_capture.release()
                logging.info("Video capture released.")
            
            if hasattr(self, 'audio_stream'):
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                logging.info("Audio input stream closed.")
            
            if hasattr(self, 'audio_output'):
                self.audio_output.stop_stream()
                self.audio_output.close()
                logging.info("Audio output stream closed.")
            
            if hasattr(self, 'audio'):
                self.audio.terminate()
                logging.info("PyAudio terminated.")
            
            logging.info("Fractal Being stopped.")
            
        except Exception as e:
            logging.error(f"Error stopping FractalBeing: {e}")

    def save_state(self, path: str):
        try:
            torch.save(self.brain.get_state_dict_custom(), path)
            logging.info(f"Brain state saved to {path}.")
        except Exception as e:
            logging.error(f"Error saving state: {e}")
        
    def load_state(self, path: str):
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.brain.load_state_dict_custom(state_dict)
            logging.info(f"Brain state loaded from {path}.")
        except Exception as e:
            logging.error(f"Error loading state: {e}")

    def update_resolution(self, width: int, height: int):
        """Update resolution and reconfigure neural networks"""
        try:
            # Store old brain state if needed
            old_state = self.brain.get_state_dict_custom() if hasattr(self.brain, 'get_state_dict_custom') else None
            
            # Update config
            self.config.video_width = width
            self.config.video_height = height
            # feature_dim is a property, so no need to set it explicitly
            
            # Reinitialize video capture with new resolution
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.video_width)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.video_height)
            
            # Reinitialize spectral and brain with new config
            self.spectral = SpectralProcessor(self.config)
            self.brain = FractalBrain(config=self.config)
            
            # Attempt to restore compatible parts of old state
            if old_state is not None:
                try:
                    self.brain.load_state_dict_custom(old_state)
                except Exception as e:
                    logging.warning(f"Could not restore previous brain state after resolution change: {e}")
            
            logging.info(f"Updated resolution to {width}x{height}, new feature dim: {self.config.feature_dim}")
            
        except Exception as e:
            logging.error(f"Error updating resolution: {e}")

class FractalBeingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Fractal Being")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.config = BeingConfig()
        self.being = FractalBeing(self.config)
        
        self.setup_ui()
        
        # Queue for updating the plot
        self.plot_queue = queue.Queue()
        
        # Start the plot updater
        self.root.after(100, self.update_plot)
    
    def setup_ui(self):
        # Control Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.start_button = ttk.Button(control_frame, text="🌟 Start", command=self.start_being)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="😴 Stop", command=self.stop_being, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="💾 Save State", command=self.save_state, state='disabled')
        self.save_button.grid(row=0, column=2, padx=5)
        
        self.load_button = ttk.Button(control_frame, text="📂 Load State", command=self.load_state, state='disabled')
        self.load_button.grid(row=0, column=3, padx=5)
        
        # Video Display
        video_frame = ttk.LabelFrame(self.root, text="👁️ Being's Vision")
        video_frame.pack(padx=10, pady=10)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        
        # Neural Activity Plot
        plot_frame = ttk.LabelFrame(self.root, text="📊 Neural Activity")
        plot_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(5,3))
        self.ax.set_title("Neural Activity")
        self.ax.set_xlabel("Neuron")
        self.ax.set_ylabel("Activation")
        self.bar_container = self.ax.bar([], [])
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
    
    def start_being(self):
        try:
            self.being.start()
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.update_video()
            messagebox.showinfo("Info", "Fractal Being started successfully.")
        except Exception as e:
            logging.error(f"Error starting being: {e}")
            messagebox.showerror("Error", f"Failed to start Fractal Being: {e}")
    
    def stop_being(self):
        try:
            self.being.stop()
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.save_button.config(state='disabled')
            self.load_button.config(state='disabled')
            self.video_label.config(image='')
            self.ax.clear()
            self.ax.set_title("Neural Activity")
            self.ax.set_xlabel("Neuron")
            self.ax.set_ylabel("Activation")
            self.canvas.draw()
            messagebox.showinfo("Info", "Fractal Being stopped successfully.")
        except Exception as e:
            logging.error(f"Error stopping being: {e}")
            messagebox.showerror("Error", f"Failed to stop Fractal Being: {e}")
    
    def save_state(self):
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
            if filepath:
                self.being.save_state(filepath)
                logging.info(f"State saved to {filepath}.")
                messagebox.showinfo("Info", f"State saved to {filepath}.")
        except Exception as e:
            logging.error(f"Error saving state: {e}")
            messagebox.showerror("Error", f"Failed to save state: {e}")
    
    def load_state(self):
        try:
            filepath = filedialog.askopenfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
            )
            if filepath:
                self.being.load_state(filepath)
                logging.info(f"State loaded from {filepath}.")
                messagebox.showinfo("Info", f"State loaded from {filepath}.")
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            messagebox.showerror("Error", f"Failed to load state: {e}")
    
    def update_video(self):
        try:
            if self.being.running and not self.being.output_queue.empty():
                video_frame, _ = self.being.output_queue.get()
                
                if len(video_frame.shape) == 2:
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2RGB)
                
                # Removed the overlay with green circles
                # overlay = self.create_visualization_overlay(video_frame)
                # output = cv2.addWeighted(video_frame, 0.7, overlay, 0.3, 0)
                output = video_frame  # Directly use the video frame without overlay
                
                image = Image.fromarray(output)
                image = image.resize((320, 240))  # Resize for better visibility
                imgtk = ImageTk.PhotoImage(image=image)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Schedule neural activity update
                activities = self.extract_activities()
                self.root.after(100, lambda: self.update_plot_data(activities))
            
            self.root.after(int(1000 / self.config.update_rate), self.update_video)  # Update based on update_rate
        except Exception as e:
            logging.error(f"Error updating video: {e}")

    def extract_activities(self):
        try:
            activities = []
            for layer in [self.being.brain.sensory, self.being.brain.processor, self.being.brain.generator]:
                # Collect activation states from the current layer
                activities.append(layer.activation_state)
                for neuron in layer.child_neurons:
                    activities.append(neuron.activation_state)
            return activities
        except Exception as e:
            logging.error(f"Error extracting activities: {e}")
            return []
    
    def update_plot_data(self, activities):
        """Update plot data asynchronously."""
        try:
            if not activities:
                logging.debug("No activities to plot.")
                self.ax.clear()
                self.ax.set_title("Neural Activity")
                self.ax.set_xlabel("Neuron")
                self.ax.set_ylabel("Activation")
                self.canvas.draw()
                return
            
            self.ax.clear()
            self.ax.set_title("Neural Activity")
            self.ax.set_xlabel("Neuron")
            self.ax.set_ylabel("Activation")
            self.ax.set_ylim(0, 1)  # Assuming activation_state is normalized between 0 and 1
            self.ax.bar(range(len(activities)), activities, color='blue')
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error updating plot: {e}")
    
    def create_visualization_overlay(self, frame: np.ndarray) -> np.ndarray:
        try:
            overlay = np.zeros_like(frame, dtype=np.uint8)
            
            if self.being:
                activities = []
                # Collect activation states from all layers
                for layer in [self.being.brain.sensory, 
                              self.being.brain.processor,
                              self.being.brain.generator]:
                    for neuron in layer.child_neurons:
                        activities.append(neuron.activation_state)

                if activities:
                    activities = np.array(activities)
                    activities = (activities - activities.min()) / (activities.max() - activities.min() + 1e-6)
                    
                    height, width = frame.shape[:2]
                    num_circles = len(activities)
                    for i, activity in enumerate(activities):
                        angle = 2 * np.pi * (i / num_circles)
                        radius = int(activity * 30)  # Increased radius for visibility
                        x = int(width / 2 + (width / 4) * np.cos(angle))
                        y = int(height / 2 + (height / 4) * np.sin(angle))
                        color = (0, 255, 0)  # Green circles
                        cv2.circle(overlay, (x, y), radius, color, -1)
            
            return overlay
        except Exception as e:
            logging.error(f"Error creating visualization overlay: {e}")
            return np.zeros_like(frame, dtype=np.uint8)
    
    def update_plot(self):
        # Placeholder for future enhancements or real-time updates
        self.root.after(1000, self.update_plot)
    
    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.being.stop()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = FractalBeingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
