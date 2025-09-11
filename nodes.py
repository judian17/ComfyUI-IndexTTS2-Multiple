# nodes.py (已更新为中英双语注释并优化单人生成流程)

import torch
import os
import ast
import time
import re # 为优化输入逻辑而添加

from . import indextts2_pipeline as pipeline
from . import utils

class IndexTTS2_Dialogue_Studio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "speaker_map": ("STRING", {
                    "multiline": True, "default": "[1]: speaker_1.wav\n[2]: speaker_2.wav",
                    "tooltip": "说话人映射表 (Speaker Map)\n将说话人ID映射到 `models/TTS/speakers/`中的音频文件 (Maps speaker IDs to audio files in `models/TTS/speakers/`).\n格式 (Format): `[ID]: filename.wav` 每行一个 (one per line).\n单人生成时可省略 `[1]:` (For single speaker, `[1]:` is optional)."
                }),
                "script": ("STRING", {
                    "multiline": True, "default": "[1]: This is the first speaker, with a neutral tone.\n[2]: And this is the second speaker, sounding a bit more excited!",
                    "tooltip": "对话脚本 (Script)\n对话内容 (The dialogue content).\n格式 (Format): `[ID] Text content` 每行一句 (one per line). 注意: 没有冒号 (Note: No colon).\n单人生成时可省略 `[1]` (For single speaker, `[1]` is optional)."
                }),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "model_unload_strategy": (["Staged (Balanced VRAM)", "Ultimate (Lowest VRAM)", "No Unloading"], {
                    "default": "Staged (Balanced VRAM)",
                    "tooltip": "模型卸载策略 (Model Unload Strategy)\nNo Unloading: 最快,最高VRAM (Fastest, highest VRAM).\nStaged: 均衡VRAM与速度 (Balanced VRAM & speed).\nUltimate: 最低VRAM,最慢 (Lowest VRAM, slowest)."
                }),
                "top_k": ("INT", {"default": 30, "min": 1, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 100}),
                "force_recache_prompts": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "emotion_audio_map": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "情感参考音频 (Emotion Audio Map)\n(可选/Optional) 将ID映射到情感参考音频 (Maps ID to an emotion reference audio).\n格式 (Format): `[ID]: emotion_audio.wav` 每行一个 (one per line)."
                }),
                "emotion_text_map": ("STRING", {
                    "multiline": True, "default": "[1]: A standard, narrative voice without any strong emotion.\n[2]: Spoken with a bright and cheerful tone, full of positive energy.\n[3]: A quiet and somber voice, tinged with a hint of sadness.",
                    "tooltip": "情感描述文本 (Emotion Text Map)\n为LLM提供详细情感描述 (Provides detailed emotional descriptions for the LLM). 支持中英混合 (Supports Chinese/English mix).\n格式 (Format): `[ID]: A detailed emotional description.`"
                }),
                "emotion_vector_map": ("STRING", {
                    "multiline": True, "default": "[1]: [0,0,0,0,0,0,0,1.0]\n[2]: [0,0,0,0,0,0,0,1.0]",
                    "tooltip": "情感向量 (Emotion Vector Map)\n直接提供8维情感向量 (Provides a direct 8D emotion vector).\n格式: `[ID]: [快乐, 生气, 悲伤, 害怕, 厌恶, 忧郁, 惊讶, 平静]`\nFormat: `[ID]: [Happy, Angry, Sad, Fear, Disgusted, Melancholic, Surprised, Calm]`\n示例 (Example): `[1]: [0,0,0.5,0,0,0,0,0.5]`"
                }),
                "emotion_alpha_map": ("STRING", {
                    "multiline": True, "default": "[1]: 1.0\n[2]: 1.0",
                    "tooltip": "情感强度 (Emotion Alpha Map)\n控制情感参考音频的强度 (Controls intensity of emotion reference audio).\n1.0=默认(default), <1.0=更弱(less), >1.0=更夸张(more).\n格式 (Format): `[ID]: 1.2`"
                }),
                "use_emo_text_map": ("STRING", {
                    "multiline": True, "default": "[1]: false\n[2]: false",
                    "tooltip": "启用情感文本 (Use Emotion Text Map)\n为指定ID启用情感文本 (Enables emotion text for a speaker).\n使用 (Use) `true` 或 (or) `false`.\n格式 (Format): `[ID]: true`"
                }),
                "use_random_map": ("STRING", {
                    "multiline": True, "default": "[1]: false\n[2]: false",
                    "tooltip": "启用随机情感 (Use Random Map)\n为情感向量选择增加随机性 (Adds randomness to emotion vector selection).\n使用 (Use) `true` 或 (or) `false`.\n格式 (Format): `[ID]: true`"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "🎤MW/MW-IndexTTS"

    # --- 新增辅助函数 ---
    def _preprocess_input(self, text_input, is_script=False):
        """
        自动格式化单人生成的输入，使其与多角色的解析逻辑兼容。
        如果输入已经包含[ID]格式，则直接返回。
        否则，为maps添加"[1]: "，为script添加"[1] "。
        """
        text_input = text_input.strip()
        if not text_input:
            return ""
        
        # 检查任何行是否以[ID]模式开头。如果是，则假定其格式正确。
        if re.search(r'^\s*\[\d+\]', text_input, re.MULTILINE):
            return text_input

        # 如果未找到格式，则假定为单个说话人（ID 1）的输入。
        lines = text_input.split('\n')
        
        # 脚本格式: "[1] The text" (无冒号)
        # Map格式: "[1]: The value" (有冒号)
        formatter = "[1] {}" if is_script else "[1]: {}"
        
        processed_lines = [formatter.format(line.strip()) for line in lines if line.strip()]
        return '\n'.join(processed_lines)

    def _check_all_prompts_cached(self, speaker_map_text):
        speaker_map = utils.parse_key_value_map(speaker_map_text)
        if not speaker_map: return True
        for sid, filename in speaker_map.items():
            cache_name = os.path.splitext(os.path.basename(filename))[0] + ".pkl"
            cache_path = os.path.join(utils.TTS_CACHE_DIR, cache_name)
            if not os.path.exists(cache_path):
                print(f"    [DEBUG] Pre-check failed: Cache not found for '{filename}'. Prompt models will be loaded.")
                return False
        print("    [DEBUG] Pre-check successful: All speaker prompts are cached. Stage A model loading will be skipped.")
        return True

    def _prepare_prompts(self, p, speaker_map_text, force_recache, skip_model_load=False):
        if not skip_model_load: p.load_prompt_models()
        else: print("    [DEBUG] _prepare_prompts: Skipping model loading as instructed.")
            
        speaker_map = utils.parse_key_value_map(speaker_map_text)
        if not speaker_map: raise ValueError("Speaker map is empty or invalid.")
        prompts = {}
        for sid, filename in speaker_map.items():
            cache_name = os.path.splitext(os.path.basename(filename))[0] + ".pkl"
            cache_path = os.path.join(utils.TTS_CACHE_DIR, cache_name)
            if not force_recache:
                cached_prompt = utils.load_cache(cache_path)
                if cached_prompt:
                    prompts[sid] = cached_prompt
                    continue
            print(f"    [DEBUG] Cache missed for '{filename}'. Processing audio...")
            audio_path = os.path.join(utils.speakers_dir, filename)
            if not os.path.exists(audio_path): raise FileNotFoundError(f"Speaker audio file not found for ID [{sid}]: '{filename}' in '{utils.speakers_dir}'")
            prompt_data = p.process_speaker_audio(audio_path)
            utils.save_cache(prompt_data, cache_path)
            prompts[sid] = prompt_data
        return prompts

    def get_emo_params_for_sid(self, sid, emo_maps):
        params = {
            "emo_audio_prompt_path": os.path.join(utils.speakers_dir, emo_maps["emotion_audio_map"][sid]) if sid in emo_maps["emotion_audio_map"] else None,
            "emo_text": emo_maps["emotion_text_map"].get(sid),
            "emo_alpha": emo_maps["emotion_alpha_map"].get(sid, 1.0),
            "use_emo_text": emo_maps["use_emo_text_map"].get(sid, False),
            "use_random": emo_maps["use_random_map"].get(sid, False),
        }
        try:
            params["emo_vector"] = ast.literal_eval(emo_maps["emotion_vector_map"][sid]) if sid in emo_maps["emotion_vector_map"] else None
        except:
            params["emo_vector"] = None
        return params
        
    def generate_dialogue(self, speaker_map, script, precision, model_unload_strategy, **kwargs):
        start_time = time.time()
        print(f"\n[IndexTTS2_NODE] Execution started with strategy: {model_unload_strategy}")
        
        # --- 优化逻辑：为单人生成模式自动补全ID ---
        speaker_map = self._preprocess_input(speaker_map, is_script=False)
        script = self._preprocess_input(script, is_script=True)
        for key, value in kwargs.items():
            if key.endswith("_map") and isinstance(value, str):
                kwargs[key] = self._preprocess_input(value, is_script=False)
        # --- 优化逻辑结束 ---
        
        generation_params = {k: kwargs[k] for k in ["top_k", "top_p", "temperature", "num_beams", "max_mel_tokens", "repetition_penalty"]}
        emo_maps = {
            "emotion_audio_map": utils.parse_key_value_map(kwargs.get("emotion_audio_map", "")),
            "emotion_text_map": utils.parse_key_value_map(kwargs.get("emotion_text_map", "")),
            "emotion_vector_map": utils.parse_key_value_map(kwargs.get("emotion_vector_map", "")),
            "emotion_alpha_map": utils.parse_key_value_map(kwargs.get("emotion_alpha_map", ""), value_type=float),
            "use_emo_text_map": utils.parse_key_value_map(kwargs.get("use_emo_text_map", ""), value_type=bool),
            "use_random_map": utils.parse_key_value_map(kwargs.get("use_random_map", ""), value_type=bool)
        }
        force_recache = kwargs.get("force_recache_prompts", False)

        if pipeline.PIPELINE_CACHE is None or pipeline.PIPELINE_CACHE.precision_str != precision:
            if pipeline.PIPELINE_CACHE is not None: pipeline.PIPELINE_CACHE.unload_all_models()
            pipeline.PIPELINE_CACHE = pipeline.IndexTTS2_Pipeline(precision=precision)
        p = pipeline.PIPELINE_CACHE

        script_lines = utils.parse_script(script)
        if not script_lines: raise ValueError("Script is empty or invalid.")
        
        waveforms = []
        all_cached = not force_recache and self._check_all_prompts_cached(speaker_map)

        if model_unload_strategy == "No Unloading":
            print("[IndexTTS2_NODE] ---> Entering 'No Unloading' logic...")
            if not all_cached: p.load_prompt_models()
            p.load_generation_models()
            p.load_decoding_models()
            speaker_prompts = self._prepare_prompts(p, speaker_map, force_recache, skip_model_load=True)
        else: # Staged (Balanced VRAM) and Ultimate (Lowest VRAM)
            print(f"[IndexTTS2_NODE] ---> Entering '{model_unload_strategy}' logic...")
            speaker_prompts = self._prepare_prompts(p, speaker_map, force_recache, skip_model_load=all_cached)
            if model_unload_strategy == "Ultimate (Lowest VRAM)" and not all_cached:
                p.unload_prompt_models()
            p.load_generation_models()

        latents_cache = []
        for i, line in enumerate(script_lines):
            sid, text = line['id'], line['text']
            if sid not in speaker_prompts: raise ValueError(f"Speaker ID [{sid}] not in speaker_map.")
            
            print(f"\n[IndexTTS2_NODE] --- Preparing line {i+1}/{len(script_lines)} for speaker [{sid}] ---")
            emo_params = self.get_emo_params_for_sid(sid, emo_maps)

            log_message = f"    > Speaker [{sid}] settings:\n"
            final_emo_source = "Default (from speaker prompt)"
            if emo_params["emo_audio_prompt_path"]:
                log_message += f"    - Emotion Audio: Applied '{os.path.basename(emo_params['emo_audio_prompt_path'])}' with Alpha: {emo_params['emo_alpha']}\n"
                final_emo_source = "Audio Reference"
            if emo_params["use_emo_text"] and emo_params["emo_text"]:
                log_message += f"    - Emotion Text:  Applied '{emo_params['emo_text'][:30]}...'\n"
                final_emo_source = "Text Description"
            elif emo_params["use_emo_text"] and not emo_params["emo_text"]:
                 log_message += "    - Emotion Text:  Enabled but no text provided. Using script line for emotion.\n"
                 final_emo_source = "Text Description (from script)"
            if emo_params["emo_vector"]:
                log_message += f"    - Emotion Vector: Applied {emo_params['emo_vector']}\n"
                final_emo_source = "Emotion Vector"
            log_message += f"    >> Final Emotion Source: {final_emo_source}"
            print(log_message)
            
            all_latents_data = p.generate_latents(speaker_prompts[sid], text, emo_params, generation_params)
            latents_cache.append({"latents_data": all_latents_data, "speaker_prompt": speaker_prompts[sid]})

        if model_unload_strategy == "No Unloading":
            print("[IndexTTS2_NODE] Decoding all waveforms...")
            for i, item in enumerate(latents_cache):
                wav = p.decode_latents(item["speaker_prompt"], item["latents_data"])
                waveforms.append(wav)
            print("[IndexTTS2_NODE] <--- Exiting 'No Unloading' logic. Models remain loaded.")
        else:
            if model_unload_strategy == "Ultimate (Lowest VRAM)": p.unload_generation_models()
            print("\n[IndexTTS2_NODE] --- STAGE C: WAVEFORM DECODING ---")
            p.load_decoding_models()
            for i, item in enumerate(latents_cache):
                print(f"[IndexTTS2_NODE] Decoding waveform {i+1}/{len(latents_cache)}...")
                wav = p.decode_latents(item["speaker_prompt"], item["latents_data"])
                waveforms.append(wav)
            print("\n[IndexTTS2_NODE] --- FINAL CLEANUP ---")
            p.unload_all_models()
            pipeline.PIPELINE_CACHE = None

        final_waveform = torch.cat(waveforms, dim=1) if waveforms else torch.zeros((1,1))
        
        end_time = time.time()
        print(f"\n[IndexTTS2_NODE] Dialogue generation finished in {end_time - start_time:.2f} seconds.")
        return ({"waveform": final_waveform.unsqueeze(0).cpu(), "sample_rate": 22050},)

NODE_CLASS_MAPPINGS = { "IndexTTS2_Dialogue_Studio": IndexTTS2_Dialogue_Studio }
NODE_DISPLAY_NAME_MAPPINGS = { "IndexTTS2_Dialogue_Studio": "🎤 IndexTTS-2 Dialogue Studio" }