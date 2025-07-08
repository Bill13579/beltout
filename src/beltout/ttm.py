from functools import lru_cache
# from pathlib import Path

# import librosa
import torchaudio

import numpy as np
import torch
import torch.nn as nn
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.gen.utils.mel import mel_spectrogram

from .models.gen.xvector import CAMPPlus

from .models.gen.decoder import ConditionalDecoder
from .models.gen.flow_matching import CausalConditionalCFM
from .models.gen.configs import CFM_PARAMS

from .models.gen.f0_predictor import ConvRNNF0Predictor
from .models.gen.hifigan import HiFTGenerator

from .models.vmvmt import PitchMvmtEncoder

from .models.gen.transformer.upsample_encoder import UpsampleConformerEncoder
from .models.s3tokenizer import S3Tokenizer

# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return torchaudio.transforms.Resample(src_sr, dst_sr).to(device)

REPO_ID = "Bill13579/beltout"

SPK_EMBED_DIM = 192
COND_DIM = 80

GEN_SR = 24000

class Flow(nn.Module):
    def __init__(
        self,
        spk_embed_affine_layer: nn.Linear,
        input_embedding: nn.Embedding,
        encoder_proj: nn.Linear,
    ):
        super().__init__()
        self.spk_embed_affine_layer = spk_embed_affine_layer
        self.input_embedding = input_embedding
        self.encoder_proj = encoder_proj

class BeltOutTTM(nn.Module):
    def __init__(
        self,
        pitchmvmt: PitchMvmtEncoder,
        flow: Flow,
        mel2wav: HiFTGenerator,
        decoder: CausalConditionalCFM,
        speaker_encoder: CAMPPlus,
        encoder: UpsampleConformerEncoder,
        tokenizer: S3Tokenizer,
        device: str,
    ):
        super().__init__()
        self.sr = GEN_SR
        self.pitchmvmt = pitchmvmt
        self.flow = flow
        self.mel2wav = mel2wav
        self.decoder = decoder
        self.speaker_encoder = speaker_encoder
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = device
        self.mel_extractor = mel_spectrogram

    @classmethod
    def from_local(cls, decoder_ckpt_path, pitchmvmt_ckpt_path, encoder_ckpt_path, flow_ckpt_path, mel2wav_ckpt_path, speaker_encoder_ckpt_path, tokenizer_ckpt_path, device, eval=True) -> 'BeltOutTTM':
        pitchmvmt = PitchMvmtEncoder()
        pitchmvmt.load_state_dict(
            load_file(pitchmvmt_ckpt_path), strict=False
        )

        # Grab bag of small layers that may or may not be used based on mode.
        spk_embed_affine_layer = nn.Linear(SPK_EMBED_DIM, COND_DIM)
        input_embedding = nn.Embedding(6561, 512)
        encoder_proj = nn.Linear(512, COND_DIM)
        flow = Flow(spk_embed_affine_layer=spk_embed_affine_layer, input_embedding=input_embedding, encoder_proj=encoder_proj)
        flow.load_state_dict(
            load_file(flow_ckpt_path), strict=False
        )
        flow.load_state_dict(
            load_file(encoder_ckpt_path), strict=False
        )

        f0_predictor = ConvRNNF0Predictor()
        mel2wav = HiFTGenerator(
            sampling_rate=GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )
        mel2wav.load_state_dict(
            load_file(mel2wav_ckpt_path), strict=False
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
        )
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=CFM_PARAMS,
            estimator=estimator,
        )
        decoder.load_state_dict(
            load_file(decoder_ckpt_path), strict=False
        )

        speaker_encoder = CAMPPlus()  # use default args
        speaker_encoder.load_state_dict(
            load_file(speaker_encoder_ckpt_path), strict=False
        )
        
        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )
        encoder.load_state_dict(
            load_file(encoder_ckpt_path), strict=False
        )

        tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        tokenizer.load_state_dict(
            load_file(tokenizer_ckpt_path), strict=False
        )

        model = cls(pitchmvmt, flow, mel2wav, decoder, speaker_encoder, encoder, tokenizer, device)
        if eval:
            model.to(device).eval()
        else:
            model.to(device).train()

        return model

    # @classmethod
    # def from_pretrained_hf(cls, decoder_ckpt_path, pitchmvmt_ckpt_path, encoder_ckpt_path, flow_ckpt_path, mel2wav_ckpt_path, speaker_encoder_ckpt_path, tokenizer_ckpt_path, local_dir=None, device="cpu") -> 'BeltOutTTM':
    #     # Check if MPS is available on macOS
    #     if device == "mps" and not torch.backends.mps.is_available():
    #         if not torch.backends.mps.is_built():
    #             print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    #         else:
    #             print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
    #         device = "cpu"
        
    #     local_paths = []
    #     for fpath in [decoder_ckpt_path, pitchmvmt_ckpt_path, encoder_ckpt_path, flow_ckpt_path, mel2wav_ckpt_path, speaker_encoder_ckpt_path, tokenizer_ckpt_path]:
    #         local_paths.append(Path(hf_hub_download(repo_id=REPO_ID, filename=fpath, local_dir=local_dir)))
    #     [decoder_ckpt_path, pitchmvmt_ckpt_path, encoder_ckpt_path, flow_ckpt_path, mel2wav_ckpt_path, speaker_encoder_ckpt_path, tokenizer_ckpt_path] = local_paths

    #     return cls.from_local(decoder_ckpt_path, pitchmvmt_ckpt_path, encoder_ckpt_path, flow_ckpt_path, mel2wav_ckpt_path, speaker_encoder_ckpt_path, tokenizer_ckpt_path, device)

    def embed_ref_x_vector(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
    ):
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: cosydec received ref longer than 10s")

        # Resample to 16kHz
        ref_wav_16 = get_resampler(ref_sr, 16000, device)(ref_wav).to(device)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        return ref_x_vector