\# ComfyUI IndexTTS2



这是一个基于 \[comfyui-indextts](https://github.com/billwuhao/ComfyUI\_IndexTTS) (billwuhao/ComfyUI\_IndexTTS: IndexTTS Voice Cloning: Supports two-person dialogue) 修改而来的 ComfyUI IndexTTS2 节点，实现了不限人数的多人语音生成。



此版本支持当前最新的 `transformers==4.38.2`，提供了更灵活的模型管理方式，并且在 6GB 显存的设备上即可运行。



\## 模型路径与依赖安装



关于模型路径的配置、音频路径的设置以及特殊依赖的安装方法，请参考原项目 \[comfyui-indextts](https://github.com/billwuhao/ComfyUI\_IndexTTS) 的说明。



\*\*请不要忘记为原作者 \[billwuhao](https://github.com/billwuhao) 点星！\*\*



\## 使用说明



多人语音生成功能通过多行文本输入来实现，每一行分别代表一个角色的音频、台词和相关参数。



具体的使用方法和参数配置，请参考项目中的示例工作流 `\\workflow-examples\\indextts2.json`，其中包含了详细的注释和节点说明。



\### 工作流示例



!\[Workflow Example](\\images\\workflow.png)

