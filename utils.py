# utils.py

import os
import re
import folder_paths
import pickle
import torch

# --- 路径定义 ---
# 采用与MegaTTS3一致的目录结构
models_dir = folder_paths.models_dir
speakers_dir = os.path.join(models_dir, "TTS", "speakers")
TTS_CACHE_DIR = os.path.join(speakers_dir, "indextts2_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)


# --- 文本解析器 ---

def parse_key_value_map(map_text, value_type=str):
    """
    解析多行文本的映射表，格式为 '[ID]: value'。
    返回一个字典 {int: value_type}。
    """
    parsed_map = {}
    pattern = re.compile(r'\[(\d+)\]\s*:\s*(.*)')
    lines = map_text.strip().split('\n')
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            key = int(match.group(1))
            raw_value = match.group(2).strip()
            try:
                # 根据指定的类型转换值
                if value_type == bool:
                    val = raw_value.lower() in ['true', '1', 'yes', 'on']
                else:
                    val = value_type(raw_value)
                parsed_map[key] = val
            except (ValueError, TypeError) as e:
                print(f"[IndexTTS2] Warning: 无法解析键 [{key}] 的值 '{raw_value}'。已跳过。错误: {e}")
    return parsed_map

def parse_script(script_text):
    """
    解析对话脚本，格式为 '[ID] text'。
    返回一个字典列表 [{'id': int, 'text': str}]。
    """
    parsed_script = []
    pattern = re.compile(r'\[(\d+)\]\s*(.*)')
    lines = script_text.strip().split('\n')
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            speaker_id = int(match.group(1))
            text = match.group(2).strip()
            if text:
                parsed_script.append({'id': speaker_id, 'text': text})
    return parsed_script


# --- 缓存工具 ---

def load_cache(cache_path):
    """从pickle文件加载缓存。"""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[IndexTTS2] Warning: 加载缓存文件 {os.path.basename(cache_path)} 失败。错误: {e}")
        return None

def save_cache(data, cache_path):
    """将数据保存到pickle文件。"""
    try:
        # 在保存前确保所有Tensor都在CPU上
        cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        with open(cache_path, 'wb') as f:
            pickle.dump(cpu_data, f)
        print(f"[IndexTTS2] 已保存音色缓存: {os.path.basename(cache_path)}")
    except Exception as e:
        print(f"[IndexTTS2] Warning: 保存缓存文件 {os.path.basename(cache_path)} 失败。错误: {e}")