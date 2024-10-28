Being

A deepdive video: https://www.youtube.com/watch?v=tJBrr-G89CY

FractalBeing is an innovative, real-time audiovisual AI system that captures live audio and video input, processes them through a sophisticated neural network architecture, and generates dynamic audiovisual feedback. This system not only interacts with its environment but also visualizes its internal neural activities, offering a glimpse into the intricate processes that drive its behavior.

Table of Contents

Introduction
FractalBeing embodies the fusion of real-time data processing, neural computation, and interactive visualization. Inspired by fractal geometry and complex systems, it leverages a multi-layered neural architecture to interpret and respond to its sensory inputs, creating a self-organizing and adaptive entity. Whether used for research, education, or as an art installation, FractalBeing offers a unique platform to explore the intersections of AI, neuroscience, and multimedia.

Features

Real-Time Audio and Video Capture: Utilizes your device's webcam and microphone to gather live audiovisual data.

Advanced Spectral Processing: Converts raw audio and video inputs into meaningful feature representations using spectral analysis.

Fractal Neural Architecture: Implements a recursive, multi-depth neural network that adapts and evolves based on input data.

Dynamic Feedback Loop: Generates synthesized audio and visual outputs that feed back into the system, creating an interactive experience.

Interactive Visualization: Displays live video with overlayed neural activity indicators and real-time plots of neural activations.

State Persistence: Allows saving and loading the neural network's state for continuity across sessions.

User-Friendly GUI: Provides an intuitive interface with controls to start, stop, and manage the FractalBeing.

Architecture

FractalBeing's architecture is designed to handle the complexities of real-time data processing and neural computation. It comprises several interconnected components that work seamlessly to deliver an interactive and responsive experience.

Data Capture

Video Capture: Uses OpenCV (cv2.VideoCapture) to access the device's webcam, capturing frames at a configurable resolution and frame rate.

The default webcam is at 0 you can change this under: Def setup_av_capture

Audio Capture: Utilizes PyAudio to stream audio data from the microphone, processing audio windows with specified sizes and hop lengths.

Spectral Processing

Audio Processing:

Short-Time Fourier Transform (STFT): Breaks down audio signals into frequency components over time.

Mel Filterbank: Applies a Mel-scale filter to the STFT magnitudes to extract perceptually relevant features.

Normalization: Logs and normalizes the Mel features to ensure consistent input for the neural network.

Video Processing:

Grayscale Conversion: Converts color frames to grayscale to reduce computational complexity.

Adaptive Resizing: Adjusts the frame size based on the system's feature dimensions.

Discrete Cosine Transform (DCT): Transforms spatial pixel data into frequency components.

Feature Extraction: Extracts significant DCT coefficients as features for the neural network.

Neural Network

FractalNeuron: The core building block, implementing a recursive neural module with adaptive scaling and depth. Each neuron can spawn child neurons up to a maximum depth, allowing for complex, fractal-like 
processing.

You can set the amount of neuronal layers (the more the harder the software is to run) and max amount of neurons in the beginning of the script under neuronal settings. 

FractalBrain: Comprises interconnected FractalNeurons arranged into sensory, processing, and generator pathways. It handles feature compression and expansion, memory buffering, and state management.

FractalDimensionAdapter: Facilitates dimension adaptation within the neural network, allowing for flexible scaling and multi-scale processing of input features.

Feedback Mechanism

Audio Output: Synthesizes audio based on the neural network's output, resampling and clipping it before playback through the device's speakers. This will 
change according to your input to the microphone, the soudn is fed to the being and it will change its tone and audio patterns. 

Video Output: Displays processed video frames with overlayed visualizations representing neural activity. This creates a visual feedback loop that reflects the system's internal states.

Visualization

Visualization plays a pivotal role in FractalBeing, offering both a window into its sensory inputs and its internal neural dynamics.

Video Display

Live Video Feed: Shows the real-time video captured from the webcam, processed and resized according to the system's configuration. This offers us a window to beings brain. 

Visualization Overlay: Superimposes dynamic graphical elements (e.g., circles) on the video feed. These elements represent the activation states of neurons, their positions, and their activities, creating a visually engaging representation of neural processes.

Neural Activity Plot

Real-Time Plotting: Displays a bar chart of neural activations, updating in real-time to reflect the ongoing processes within the FractalBrain. These may freeze so if someone wants to fix them. 
Please do go ahead. 

Interactivity: Users can observe how different neurons activate in response to varying inputs, providing insights into the system's decision-making and adaptation mechanisms.

Usage

Installation

Ensure that you have Python 3.7 or higher installed on your system. Follow these steps to set up FractalBeing:

Clone the Repository

git clone https://github.com/anttiluode/being.git

Create a Virtual Environment (Optional but Recommended)

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python app.py

Upon launching, the GUI provides several controls to manage FractalBeing's operations:

🌟 Start: Initiates the audiovisual capture and processing, activating the neural network and feedback loops.

😴 Stop: Halts all operations, releasing hardware resources and stopping audio playback.

💾 Save State: Saves the current state of the neural network, including memory buffers and configuration settings, to a file.

📂 Load State: Loads a previously saved neural network state from a file, restoring its memory and configurations.

Technical Details

Dependencies

torch
numpy
opencv-python-headless
pyaudio
matplotlib
tkinter
pillow
scipy
dataclasses

If anything is missing. Sorry. 

FractalBeing relies on several Python libraries to function seamlessly:

Core Libraries:

torch: For building and managing neural networks.
numpy: For numerical computations.
opencv-python (cv2): For video capture and processing.
pyaudio: For audio capture and playback.
matplotlib: For plotting neural activity.
tkinter: For the graphical user interface.
Pillow (PIL): For image processing within Tkinter.

Additional Libraries:

scipy: For signal processing tasks like STFT and resampling.
dataclasses: For structured configuration management.
logging: For detailed logging of system operations and errors.
Configuration
FractalBeing is highly configurable, allowing users to adjust various parameters to tailor its behavior:

Video Settings:

video_width: Width of the video frame (default: 160).
video_height: Height of the video frame (default: 120).
fps: Frames per second for video capture (default: 15).

Audio Settings:

sample_rate: Audio sampling rate in Hz (default: 44100).
audio_window: Size of the audio window for processing (default: 1024).
audio_hop: Hop length for audio processing (default: 512).
Neural Settings:

hidden_dim: Dimensionality of the hidden layers in the neural network (default: 128).
max_neurons: Maximum number of neurons (default: 1000).
max_depth: Maximum depth for recursive neural structures (default: 4).

System Settings:

feedback_strength: Strength of the feedback loop (default: 0.1).
update_rate: Rate at which the GUI updates in Hz (default: 30).
Users can modify these settings by adjusting the BeingConfig dataclass within the code or by implementing a configuration file interface for more flexibility.


Please ensure that your contributions adhere to the project's coding standards and include appropriate tests where necessary.

License
This project is licensed under the MIT License.

Acknowledgments

Antorphic: Without Claude 3.5 Sonnet this project would not have been feasible. 
OpenAI: I was passing notes between Claude and different versions ChatGPT to do this
PyTorch Community: For their robust and flexible deep learning framework.
OpenCV Contributors: For the powerful computer vision tools utilized in video processing.
Matplotlib and Tkinter Developers: For enabling rich data visualization and user interfaces.

Future Work

If anyone wants to take this "ball" and develop being further. Please go ahead. 

ChatGPT was thinking of futher developing these things: 

Enhanced Neural Architecture: Integrate more advanced neural modules and learning algorithms to improve adaptability and responsiveness.
Extended Visualization Options: Incorporate 3D visualizations and more interactive plots to provide deeper insights into neural activities.
User Configuration Interface: Develop a settings panel within the GUI to allow users to adjust configurations without modifying the code.
Multi-Modal Inputs: Expand beyond audio and video to include other sensory data like temperature, motion, or textual inputs.
Deployment Options: Optimize FractalBeing for deployment on various platforms, including mobile devices and embedded systems.
Your contributions and ideas are highly encouraged to help realize these future enhancements!

Antti
