# nodes.py (已更新为中英双语注释、优化单人生成流程、增加SRT字幕输出并集成最终版智能断句和多行解析修正)

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
                    "multiline": True, "default": "[1]: speaker_1.wav",
                    "tooltip": "说话人映射表 (Speaker Map)\n将说话人ID映射到 `models/TTS/speakers/`中的音频文件。(Maps speaker IDs to audio files in `models/TTS/speakers/`)."
                }),
                "script": ("STRING", {
                    "multiline": True, "default": "[1]: This is a test sentence.\nThis is a second line for the same speaker.",
                    "tooltip": "对话脚本 (Script)\n支持单个角色多行文本，换行符会强制作为字幕切分点。(Multi-line text for a single speaker is supported. Newlines will force subtitle breaks.)"
                }),
                "subtitle_max_length": ("INT", {
                    "default": 30, "min": 5, "max": 100,
                    "tooltip": "字幕单行最大长度 (Subtitle Max Length)\n用于智能断句。中文字符计为1，英文单词计为1。(For smart splitting. A Chinese char counts as 1, an English word counts as 1.)"
                }),
                "silence_duration": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.05,
                    "tooltip": "句间静音时长(秒)\n在每两段生成的音频之间插入指定时长的静音。(Silence duration in seconds between each generated audio clip.)"
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
                "emotion_audio_map": ("STRING", {"multiline": True, "default": ""}),
                "emotion_text_map": ("STRING", {"multiline": True, "default": ""}),
                "emotion_vector_map": ("STRING", {"multiline": True, "default": ""}),
                "emotion_alpha_map": ("STRING", {"multiline": True, "default": ""}),
                "use_emo_text_map": ("STRING", {"multiline": True, "default": ""}),
                "use_random_map": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "srt_subtitles",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "🎤MW/MW-IndexTTS"

    def _get_mixed_length(self, text):
        han_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
        word_count = len(re.findall(r'\b[a-zA-Z]+\b', text))
        return han_count + word_count

    def _split_long_sentence(self, sentence, max_len):
        if self._get_mixed_length(sentence) <= max_len:
            return [sentence]
        parts = re.split(r'([,，、\s]+)', sentence)
        chunks, current_chunk = [], ""
        for part in parts:
            if not part: continue
            if current_chunk and self._get_mixed_length(current_chunk + part) > max_len:
                chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += part
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def _split_text_for_srt(self, text, max_len=15):
        if not text.strip(): return []
        lines = text.split('\n')
        all_chunks = []
        for line in lines:
            if not line.strip(): continue
            parts = re.split(r'([.?!。？！])', line)
            sentences = ["".join(i).strip() for i in zip(parts[0::2], parts[1::2])]
            if len(parts[0::2]) > len(parts[1::2]):
                sentences.append(parts[-1].strip())
            sentences = [s for s in sentences if s]
            if not sentences: continue
            temp_chunk = sentences[0]
            for i in range(1, len(sentences)):
                if self._get_mixed_length(temp_chunk + " " + sentences[i]) <= max_len:
                    temp_chunk += " " + sentences[i]
                else:
                    all_chunks.extend(self._split_long_sentence(temp_chunk, max_len))
                    temp_chunk = sentences[i]
            all_chunks.extend(self._split_long_sentence(temp_chunk, max_len))
        return all_chunks

    def _format_srt_time(self, seconds):
        millis = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _generate_srt(self, srt_data):
        srt_content = []
        for i, data in enumerate(srt_data):
            index = i + 1
            start_time_str = self._format_srt_time(data['start'])
            end_time_str = self._format_srt_time(data['end'])
            text = data['text']
            srt_content.append(f"{index}\n{start_time_str} --> {end_time_str}\n{text}\n")
        return "\n".join(srt_content)

    def _preprocess_input(self, text_input, is_script=False):
        text_input = text_input.strip()
        if not text_input: return ""
        if not is_script and re.search(r'^\s*\[\d+\]', text_input, re.MULTILINE): return text_input
        lines = text_input.split('\n')
        if is_script and any(re.match(r'^\s*\[\d+\]', line) for line in lines):
             return text_input
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
                print(f"    [DEBUG] Pre-check failed: Cache not found for '{filename}'.")
                return False
        print("    [DEBUG] Pre-check successful: All speaker prompts are cached.")
        return True

    def _prepare_prompts(self, p, speaker_map_text, force_recache, skip_model_load=False):
        if not skip_model_load: p.load_prompt_models()
        speaker_map = utils.parse_key_value_map(speaker_map_text)
        if not speaker_map: raise ValueError("Speaker map is empty or invalid.")
        prompts = {}
        for sid, filename in speaker_map.items():
            cache_name = os.path.splitext(os.path.basename(filename))[0] + ".pkl"
            cache_path = os.path.join(utils.TTS_CACHE_DIR, cache_name)
            if not force_recache and (cached_prompt := utils.load_cache(cache_path)):
                prompts[sid] = cached_prompt
                continue
            audio_path = os.path.join(utils.speakers_dir, filename)
            if not os.path.exists(audio_path): raise FileNotFoundError(f"Speaker audio file not found for ID [{sid}]: '{filename}' in '{utils.speakers_dir}'")
            prompt_data = p.process_speaker_audio(audio_path)
            utils.save_cache(prompt_data, cache_path)
            prompts[sid] = prompt_data
        return prompts

    def get_emo_params_for_sid(self, sid, emo_maps):
        params = {"emo_text": emo_maps["emotion_text_map"].get(sid), "emo_alpha": emo_maps["emotion_alpha_map"].get(sid, 1.0), "use_emo_text": emo_maps["use_emo_text_map"].get(sid, False), "use_random": emo_maps["use_random_map"].get(sid, False)}
        if sid in emo_maps["emotion_audio_map"]: params["emo_audio_prompt_path"] = os.path.join(utils.speakers_dir, emo_maps["emotion_audio_map"][sid]) 
        else: params["emo_audio_prompt_path"] = None
        try: params["emo_vector"] = ast.literal_eval(emo_maps["emotion_vector_map"][sid]) if sid in emo_maps["emotion_vector_map"] else None
        except: params["emo_vector"] = None
        return params

    def _process_decoding_and_srt(self, p, item, original_text, subtitle_max_length):
        wav = p.decode_latents(item["speaker_prompt"], item["latents_data"])
        total_duration = wav.shape[1] / 22050.0
        sub_texts = self._split_text_for_srt(original_text, subtitle_max_length)
        sub_srt_data = []
        if not sub_texts: return wav, sub_srt_data
        total_text_length = sum(self._get_mixed_length(t) for t in sub_texts)
        if total_text_length == 0: total_text_length = 1
        local_cumulative_duration = 0.0
        for sub_text in sub_texts:
            sub_len = self._get_mixed_length(sub_text)
            sub_duration = total_duration * (sub_len / total_text_length)
            start_time_sec = local_cumulative_duration
            end_time_sec = local_cumulative_duration + sub_duration
            sub_srt_data.append({'start': start_time_sec, 'end': end_time_sec, 'text': sub_text})
            local_cumulative_duration = end_time_sec
        return wav, sub_srt_data

    def generate_dialogue(self, speaker_map, script, subtitle_max_length, silence_duration, precision, model_unload_strategy, **kwargs):
        start_time = time.time()
        print(f"\n[IndexTTS2_NODE] Execution started with strategy: {model_unload_strategy}")
        
        speaker_map = self._preprocess_input(speaker_map, is_script=False)
        script_with_prefix = self._preprocess_input(script, is_script=True)

        script_lines_for_parsing = []
        current_id, current_text = None, []
        for line in script_with_prefix.strip().split('\n'):
            match = re.match(r'^\s*\[(\d+)\]\s*(.*)', line)
            if match:
                if current_id is not None:
                    script_lines_for_parsing.append(f"[{current_id}] {'\n'.join(current_text)}")
                current_id, current_text = match.group(1), [match.group(2)]
            elif current_id is not None:
                current_text.append(line)
        if current_id is not None:
            script_lines_for_parsing.append(f"[{current_id}] {'\n'.join(current_text)}")
        final_script_to_parse = "\n".join(script_lines_for_parsing)

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

        script_lines = utils.parse_script(final_script_to_parse)
        if not script_lines: raise ValueError("Script is empty or invalid.")
        
        waveforms, srt_data, cumulative_duration = [], [], 0.0
        all_cached = not force_recache and self._check_all_prompts_cached(speaker_map)

        if model_unload_strategy == "No Unloading":
            if not all_cached: p.load_prompt_models()
            p.load_generation_models()
            p.load_decoding_models()
            speaker_prompts = self._prepare_prompts(p, speaker_map, force_recache, skip_model_load=True)
        else:
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
            all_latents_data = p.generate_latents(speaker_prompts[sid], text, emo_params, generation_params)
            latents_cache.append({"latents_data": all_latents_data, "speaker_prompt": speaker_prompts[sid]})

        decoding_logic = (lambda i, item: 
            self._process_decoding_and_srt(p, item, script_lines[i]['text'], subtitle_max_length)
        )

        if model_unload_strategy == "No Unloading":
            print("[IndexTTS2_NODE] Decoding all waveforms...")
            for i, item in enumerate(latents_cache):
                wav, sub_srt_data = decoding_logic(i, item)
                waveforms.append(wav)
                current_line_duration = wav.shape[1] / 22050.0
                for data in sub_srt_data:
                    data['start'] += cumulative_duration
                    data['end'] += cumulative_duration
                    srt_data.append(data)
                
                cumulative_duration += current_line_duration
                if silence_duration > 0 and i < len(latents_cache) - 1:
                    cumulative_duration += silence_duration
        else:
            if model_unload_strategy == "Ultimate (Lowest VRAM)": p.unload_generation_models()
            print("\n[IndexTTS2_NODE] --- STAGE C: WAVEFORM DECODING ---")
            p.load_decoding_models()
            for i, item in enumerate(latents_cache):
                print(f"[IndexTTS2_NODE] Decoding waveform {i+1}/{len(latents_cache)}...")
                wav, sub_srt_data = decoding_logic(i, item)
                waveforms.append(wav)
                current_line_duration = wav.shape[1] / 22050.0
                for data in sub_srt_data:
                    data['start'] += cumulative_duration
                    data['end'] += cumulative_duration
                    srt_data.append(data)

                cumulative_duration += current_line_duration
                if silence_duration > 0 and i < len(latents_cache) - 1:
                    cumulative_duration += silence_duration
            
            print("\n[IndexTTS2_NODE] --- FINAL CLEANUP ---")
            p.unload_all_models()
            pipeline.PIPELINE_CACHE = None

        if silence_duration > 0 and len(waveforms) > 1:
            sample_rate = 22050
            num_silent_samples = int(sample_rate * silence_duration)
            silent_wav = torch.zeros((1, num_silent_samples), device='cpu')
            waveforms_with_silence = []
            for i, wav in enumerate(waveforms):
                waveforms_with_silence.append(wav)
                if i < len(waveforms) - 1:
                    waveforms_with_silence.append(silent_wav)
            final_waveform = torch.cat(waveforms_with_silence, dim=1)
        else:
            final_waveform = torch.cat(waveforms, dim=1) if waveforms else torch.zeros((1, 1))

        srt_output = self._generate_srt(srt_data)
        
        end_time = time.time()
        print(f"\n[IndexTTS2_NODE] Dialogue generation finished in {end_time - start_time:.2f} seconds.")
        
        return ({"waveform": final_waveform.unsqueeze(0).cpu(), "sample_rate": 22050}, srt_output,)

    def _process_decoding_and_srt(self, p, item, original_text, subtitle_max_length):
        wav = p.decode_latents(item["speaker_prompt"], item["latents_data"])
        total_duration = wav.shape[1] / 22050.0
        sub_texts = self._split_text_for_srt(original_text, subtitle_max_length)
        sub_srt_data = []
        if not sub_texts: return wav, sub_srt_data
        total_text_length = sum(self._get_mixed_length(t) for t in sub_texts)
        if total_text_length == 0: total_text_length = 1
        local_cumulative_duration = 0.0
        for sub_text in sub_texts:
            sub_len = self._get_mixed_length(sub_text)
            sub_duration = total_duration * (sub_len / total_text_length)
            start_time_sec = local_cumulative_duration
            end_time_sec = local_cumulative_duration + sub_duration
            sub_srt_data.append({'start': start_time_sec, 'end': end_time_sec, 'text': sub_text})
            local_cumulative_duration = end_time_sec
        return wav, sub_srt_data

NODE_CLASS_MAPPINGS = { "IndexTTS2_Dialogue_Studio": IndexTTS2_Dialogue_Studio }
NODE_DISPLAY_NAME_MAPPINGS = { "IndexTTS2_Dialogue_Studio": "🎤 IndexTTS-2 Dialogue Studio" }