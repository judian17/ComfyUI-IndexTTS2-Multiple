[中文版说明](README-zh.md)

# ComfyUI IndexTTS2

This is the ComfyUI IndexTTS2 node, a modified version based on [comfyui-indextts](https://github.com/billwuhao/ComfyUI_IndexTTS) (billwuhao/ComfyUI_IndexTTS: IndexTTS Voice Cloning: Supports two-person dialogue), which enables multi-person speech generation without a limit on the number of speakers.

This version supports the latest `transformers==4.38.2`, offers flexible model management, and is optimized to run on devices with as little as 6GB of VRAM.

## Model Paths & Dependency Installation

For instructions on configuring model paths, setting audio paths, and installing special dependencies, please refer to the documentation of the original [comfyui-indextts](https://github.com/billwuhao/ComfyUI_IndexTTS) project.

**Please remember to star the original author's repository, [billwuhao](https://github.com/billwuhao)!**

## How to Use

Multi-person speech generation is achieved by providing multi-line text inputs, where each line represents the reference audio, script, and parameters for a specific speaker.

For detailed instructions and parameter configurations, please refer to the example workflow `\workflow-examples\indextts2.json`, which includes comprehensive annotations and node descriptions.

### Workflow Example

![Workflow Example](\images\workflow.png)
