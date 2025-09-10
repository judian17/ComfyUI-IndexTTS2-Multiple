# indextts2_pipeline.py (已添加调试信息)

# ... (imports and path definitions remain the same) ...
import os
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import warnings
import gc
from omegaconf import OmegaConf
import sys
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from transformers import AutoTokenizer, Wav2Vec2BertModel
from modelscope import AutoModelForCausalLM
import safetensors
from transformers import SeamlessM4TFeatureExtractor
import random
import torch.nn.functional as F

import folder_paths
models_dir = folder_paths.models_dir
base_tts_dir = os.path.join(models_dir, "TTS")
model_path_v2 = os.path.join(base_tts_dir, "IndexTTS-2")
w2v_bert_path = os.path.join(base_tts_dir, "w2v-bert-2.0")
bigvgan_path = os.path.join(base_tts_dir, "bigvgan_v2_22khz_80band_256x")
campplus_path = os.path.join(base_tts_dir, "campplus")
maskgct_path = os.path.join(base_tts_dir, "MaskGCT")

PIPELINE_CACHE = None
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class IndexTTS2_Pipeline:
    def __init__(self, precision='fp16'):
        print("[PIPELINE] Initializing Pipeline instance...")
        self.precision_str = precision
        self.is_fp16 = (precision == 'fp16')
        
        if torch.cuda.is_available(): self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device = "mps"; self.is_fp16 = False
        else: self.device = "cpu"; self.is_fp16 = False
        
        if self.device == "cpu": print("[PIPELINE] Warning: Running in CPU mode.")

        self.cfg = OmegaConf.load(os.path.join(model_path_v2, "config.yaml"))
        self.dtype = torch.float16 if self.is_fp16 else torch.float32

        self.gpt = self.qwen_emo = self.semantic_model = self.semantic_codec = self.s2mel = None
        self.campplus_model = self.bigvgan = self.extract_features = None
        self.emo_matrix = self.spk_matrix = self.semantic_mean = self.semantic_std = None
        
        self.normalizer = TextNormalizer()
        self.tokenizer = TextTokenizer(os.path.join(model_path_v2, self.cfg.dataset["bpe_model"]), self.normalizer)
        mel_fn_args = { "n_fft": 1024, "win_size": 1024, "hop_size": 256, "num_mels": 80, "sampling_rate": 22050, "fmin": 0, "fmax": 8000, "center": False }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

    def _unload_models(self, model_attrs):
        for attr in model_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def load_prompt_models(self):
        if self.semantic_model is not None:
            print("    [DEBUG] Stage A models already loaded. Skipping.")
            return
        print("    [DEBUG] Loading Stage A: Prompt Processing Models...")
        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(w2v_bert_path)
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_bert_path)
        stat = torch.load(os.path.join(model_path_v2, self.cfg.w2v_stat))
        self.semantic_mean = stat["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stat["var"]).to(self.device)
        self.semantic_model = self.semantic_model.to(self.device).eval()
        self.semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        ckpt = os.path.join(maskgct_path, "semantic_codec", "model.safetensors")
        safetensors.torch.load_model(self.semantic_codec, ckpt)
        self.semantic_codec = self.semantic_codec.to(self.device).eval()
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        ckpt_path = os.path.join(campplus_path, "campplus_cn_common.bin")
        self.campplus_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.campplus_model = self.campplus_model.to(self.device).eval()
        print("    [DEBUG] Stage A Models LOADED.")

    def unload_prompt_models(self):
        print("    [DEBUG] Unloading Stage A Models...")
        self._unload_models(['extract_features', 'semantic_model', 'semantic_codec', 'campplus_model', 'semantic_mean', 'semantic_std'])
        print("    [DEBUG] Stage A Models UNLOADED.")

    def load_generation_models(self):
        if self.gpt is not None:
            print("    [DEBUG] Stage B models already loaded. Skipping.")
            return
        print("    [DEBUG] Loading Stage B: Acoustic Generation Models...")
        self.qwen_emo = QwenEmotion(os.path.join(model_path_v2, self.cfg.qwen_emo_path))
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        load_checkpoint(self.gpt, os.path.join(model_path_v2, self.cfg.gpt_checkpoint))
        self.gpt = self.gpt.to(self.device)
        use_deepspeed_flag = False
        if self.is_fp16:
            try:
                import deepspeed; use_deepspeed_flag = True
            except ImportError: print("[PIPELINE] Warning: is_fp16 is True, but DeepSpeed is not installed.")
        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed_flag, kv_cache=True, half=self.is_fp16)
        if self.is_fp16: self.gpt.half()
        self.gpt.eval()
        self.emo_matrix = torch.load(os.path.join(model_path_v2, self.cfg.emo_matrix), map_location=self.device)
        self.spk_matrix = torch.load(os.path.join(model_path_v2, self.cfg.spk_matrix), map_location=self.device)
        self.emo_num = list(self.cfg.emo_num)
        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)
        print("    [DEBUG] Stage B Models LOADED.")

    def unload_generation_models(self):
        print("    [DEBUG] Unloading Stage B Models...")
        self._unload_models(['qwen_emo', 'gpt', 'emo_matrix', 'spk_matrix'])
        print("    [DEBUG] Stage B Models UNLOADED.")
        
    def load_decoding_models(self):
        if self.s2mel is not None:
            print("    [DEBUG] Stage C models already loaded. Skipping.")
            return
        print("    [DEBUG] Loading Stage C: Waveform Decoding Models...")
        if self.semantic_codec is None:
            print("    [DEBUG] semantic_codec is needed for decoding, loading it now.")
            self.semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
            ckpt = os.path.join(maskgct_path, "semantic_codec", "model.safetensors")
            safetensors.torch.load_model(self.semantic_codec, ckpt)
            self.semantic_codec = self.semantic_codec.to(self.device).eval()
        self.s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        load_checkpoint2(self.s2mel, None, os.path.join(model_path_v2, self.cfg.s2mel_checkpoint), load_only_params=True, ignore_modules=[])
        self.s2mel = self.s2mel.to(self.device).eval()
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_path).to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        print("    [DEBUG] Stage C Models LOADED.")
            
    def unload_decoding_models(self):
        print("    [DEBUG] Unloading Stage C Models...")
        self._unload_models(['s2mel', 'bigvgan', 'semantic_codec'])
        print("    [DEBUG] Stage C Models UNLOADED.")

    def load_all_models(self):
        self.load_prompt_models()
        self.load_generation_models()
        self.load_decoding_models()

    def unload_all_models(self):
        print("[PIPELINE] Unloading ALL models.")
        self.unload_prompt_models()
        self.unload_generation_models()
        self.unload_decoding_models()

    # ... The rest of the file (get_emb, process_speaker_audio, generate_latents, decode_latents, helpers) remains unchanged from the last complete version ...
    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(input_features=input_features, attention_mask=attention_mask, output_hidden_states=True)
        feat = vq_emb.hidden_states[17]
        return (feat - self.semantic_mean) / self.semantic_std

    @torch.no_grad()
    def process_speaker_audio(self, audio_path):
        # self.load_prompt_models() # This is now called from the node logic
        audio, sr = librosa.load(audio_path, sr=None)
        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio_t).to(self.device)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio_t).to(self.device)
        inputs = self.extract_features(audio_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)
        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.float())
        feat = torchaudio.compliance.kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))
        prompt_condition = self.s2mel.models['length_regulator'](S_ref, ylens=torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device), n_quantizers=3)[0]
        return {"spk_cond_emb": spk_cond_emb.cpu(), "style": style.cpu(), "prompt_condition": prompt_condition.cpu(), "ref_mel": ref_mel.cpu()}

    @torch.no_grad()
    def generate_latents(self, speaker_prompt, text, emotion_params, generation_params):
        # self.load_generation_models() # This is now called from the node logic
        spk_cond_emb = speaker_prompt["spk_cond_emb"].to(self.device)
        style = speaker_prompt["style"].to(self.device)
        
        emo_vector = emotion_params.get("emo_vector")
        if emotion_params.get("use_emo_text", False):
            emo_text_to_process = emotion_params.get("emo_text") or text
            emo_dict = self.qwen_emo.inference(emo_text_to_process)
            emo_vector = list(emo_dict.values())

        emovec = None
        if emo_vector:
            weight_vector = torch.tensor(emo_vector, device=self.device, dtype=self.dtype)
            spk_matrix_dtype = [tmp.to(self.dtype) for tmp in self.spk_matrix]
            random_index = [random.randint(0, x - 1) for x in self.emo_num] if emotion_params.get("use_random", False) else [find_most_similar_cosine(style, tmp) for tmp in spk_matrix_dtype]
            emo_matrix_selected = torch.cat([tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)], 0)
            emovec_mat = torch.sum(weight_vector.unsqueeze(1) * emo_matrix_selected, 0).unsqueeze(0)
            emovec = emovec_mat

        emo_audio_prompt_path = emotion_params.get("emo_audio_prompt_path")
        emo_cond_emb = spk_cond_emb
        if emo_audio_prompt_path:
            emo_audio, sr = librosa.load(emo_audio_prompt_path, sr=16000)
            emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
            emo_cond_emb = self.get_emb(emo_inputs["input_features"].to(self.device), emo_inputs["attention_mask"].to(self.device))
        
        all_latents_data = []
        sentences = self.tokenizer.split_sentences(self.tokenizer.tokenize(text))
        for sent_tokens in sentences:
            text_tokens = torch.tensor(self.tokenizer.convert_tokens_to_ids(sent_tokens), dtype=torch.int32, device=self.device).unsqueeze(0)
            use_speed = torch.zeros(spk_cond_emb.size(0), device=self.device, dtype=torch.long)
            
            with torch.amp.autocast(self.device, dtype=self.dtype):
                base_emovec = self.gpt.merge_emovec(spk_cond_emb, emo_cond_emb, torch.tensor([spk_cond_emb.shape[-1]], device=self.device), torch.tensor([emo_cond_emb.shape[-1]], device=self.device), alpha=emotion_params.get("emo_alpha", 1.0))
                final_emovec = base_emovec
                if emovec is not None:
                    weight_sum = torch.sum(weight_vector)
                    final_emovec = emovec + (1 - weight_sum if weight_sum <= 1 else 0) * base_emovec

                gen_params_copy = generation_params.copy()
                max_generate_length = gen_params_copy.pop("max_mel_tokens", self.gpt.max_mel_tokens)

                codes, speech_conditioning_latent = self.gpt.inference_speech(
                    spk_cond_emb, text_tokens, emo_cond_emb,
                    cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                    emo_cond_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                    emo_vec=final_emovec,
                    max_generate_length=max_generate_length,
                    do_sample=True,
                    **gen_params_copy
                )
                
                stop_mel_token = self.cfg.gpt.stop_mel_token
                code_len_stop = (codes[0] == stop_mel_token).nonzero(as_tuple=False)
                code_len = code_len_stop[0] if len(code_len_stop) > 0 else len(codes[0])
                codes = codes[:, :code_len]
                code_lens = torch.LongTensor([codes.shape[-1]]).to(self.device)

                latent = self.gpt(speech_conditioning_latent, text_tokens, torch.tensor([text_tokens.shape[-1]], device=self.device),
                    codes, code_lens, emo_cond_emb, cond_mel_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                    emo_cond_mel_lengths=torch.tensor([emo_cond_emb.shape[-1]], device=self.device), emo_vec=final_emovec, use_speed=use_speed)
                
                all_latents_data.append({'latent': latent, 'codes': codes, 'code_lens': code_lens})
        
        return all_latents_data
    
    @torch.no_grad()
    def decode_latents(self, speaker_prompt, all_latents_data):
        # self.load_decoding_models() # This is now called from the node logic
        prompt_condition = speaker_prompt["prompt_condition"].to(self.device)
        ref_mel = speaker_prompt["ref_mel"].to(self.device)
        style = speaker_prompt["style"].to(self.device)
        
        wavs = []
        for latents_data in all_latents_data:
            latent = latents_data['latent']
            codes = latents_data['codes']
            code_lens = latents_data['code_lens']
            
            with torch.amp.autocast(self.device, dtype=self.dtype):
                latent = self.s2mel.models['gpt_layer'](latent)
                S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()
                cond = self.s2mel.models['length_regulator'](S_infer, ylens=target_lengths)[0]
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = self.s2mel.models['cfm'].inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(self.device), ref_mel, style, None, n_timesteps=25, inference_cfg_rate=0.7)
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                wav = self.bigvgan(vc_target.to(torch.float32)).squeeze(1)
                wavs.append(wav.cpu())

        if not wavs: return torch.tensor([])
        final_wav = torch.cat(wavs, dim=1)
        max_abs_val = torch.max(torch.abs(final_wav))
        if max_abs_val > 0: final_wav = final_wav / max_abs_val * 0.95
        return final_wav

def find_most_similar_cosine(query_vector, matrix):
    similarities = F.cosine_similarity(query_vector.float(), matrix.float(), dim=1)
    return torch.argmax(similarities)

class QwenEmotion:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    def inference(self, text_input):
        print(f"    [DEBUG][Qwen] Analyzing text: '{text_input[:30]}...'")
        try:
            messages = [{"role": "system", "content": "文本情感分类"}, {"role": "user", "content": text_input}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512, pad_token_id=self.tokenizer.eos_token_id)
            response_text = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            content_dict = ast.literal_eval(response_text)
            if not isinstance(content_dict, dict): content_dict = {}
        except Exception as e:
            print(f"    [DEBUG][Qwen] Error during inference: {e}. Falling back to neutral.")
            content_dict = {}
        final_emo_dict = {"happy": content_dict.get("高兴", 0.0), "angry": content_dict.get("愤怒", 0.0), "sad": content_dict.get("悲伤", 0.0), "fear": content_dict.get("恐惧", 0.0), "disgusted": content_dict.get("反感", 0.0), "melancholic": content_dict.get("低落", 0.0), "surprised": content_dict.get("惊讶", 0.0), "calm": content_dict.get("自然", 0.0)}
        if all(v == 0.0 for v in final_emo_dict.values()): final_emo_dict['calm'] = 1.0
        return final_emo_dict