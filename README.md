# CrackManager v1

## Server

### 1. Requirements

* Python 3.7.6 x64
  * <https://www.python.org/downloads/release/python-376/>
* Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 x64
  * <https://aka.ms/vs/16/release/vc_redist.x64.exe>
* (Optional) Latest NVIDIA GPU Drivers
* (Optional) NVIDIA CUDA Toolkit 10.1 update2
  * <https://developer.nvidia.com/cuda-10.1-download-archive-update2>
* (Optional) NVIDIA cuDNN v7.6.5 for CUDA 10.1
  * <https://developer.nvidia.com/rdp/cudnn-download>
* (Optional) Qt Designer 5.11

### 2. Installation

``` bash
python -m pip install --upgrade pip
pip install scikit-learn
pip install opencv-python
pip install keras
pip install albumentations
pip install segmentation-models
pip install pyqt5
pip install qrcode
pip install pyftpdlib
```

### 3. Run

``` bash
python crack_manager.py
```

## Holo

### 1. Enviorments

* Microsoft HoloLens 1
* CrackManager v1
  
### 2. Run

* Air-Tap on HoloCrackManager icon on Hololens

### Build

#### 1. Build Requirements

* Windows 10 Pro 1903 x64
* Windows Software Developement Kit - Windows 10.0.18362.1
* Unity 2018.4.17f1
  * UWP Build Support (IL2CPP)
  * 3D Project
* Visual Studio Community 2019 16.4.3
  * .NET Desktop Developement
    * .NET SDK 4.7.1
  * C++ Desktop Develeopement
  * Universal Windows Platform Develeopement

#### 2. Build Settings

* Platform: Universal Windows Platform
* Target Device: HoloLens
* Architecture: x86
* Build Type: D3D
* Target SDK Version: 10.0.18362.0
* Minimum Platform Version: 10.0.10240.0
* Visual Studio Version: Visual Studio 2019
* Build and Run on: Local Machine
* Build configuration: Release

#### 3. Player Settings

* Publishing Settings
  * Packaging
    * Company Name: iCELab
    * Product Name: CrackManager
    * Version: 1.0.0.0
  * Application UI
    * Package Name: CrackManager
    * Version: 1.0.0.0
    * Description: CrackManager
  * Capabilities
    * InternetClient
    * InternetClientServer
    * PrivateNetworkClientServer
    * WebCam
    * Microphone
    * SpatialPerception
  * Supported Device Families
    * Holographic
  * XR Settings
    * Virtual Reality Supported
    * Depth Format: 16-bit depth
    * Stereo Rendering Mode: Single Pass

#### 4. Lighting (Per Scene)

* Scene
  * Realtime Lighting
    * Disable Realtime Global Illumination

#### 5. Import Unity Packages

1. Microsoft.MixedReality.Toolkit.Unity.Foundation.2.2.0.unitypackage
2. Microsoft.MixedReality.Toolkit.Unity.Extensions.2.2.0.unitypackage
3. Microsoft.MixedReality.Toolkit.Unity.Tools.2.2.0.unitypackage
4. Microsoft.MixedReality.Toolkit.Unity.Examples.2.2.0.unitypackage
5. Plugins.FileManagement.unitypackage
6. Plugins.IngameDebugConsole.unitypackage
7. OpenCVforUnity (Unity Store)
8. CrackManager.unitypackage

#### 6. Build Packages

1. Mixed Reality Toolkit - Utilities - Build Window - Build Unity Project
2. Mixed Reality Toolkit - Utilities - Build Window - Open in Visual Studio
3. Solution: Release x86 Remote Computer
4. Set as startup project: CrackManager(Universal Windows)
5. Build without debug
