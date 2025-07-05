from pathlib import Path

import librosa
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.pitchmvmt import PitchMvmtEncoder


REPO_ID = "Bill13579/beltout"


class BeltOutTTM(nn.Module):
    def __init__(
        self,
        s3gen: S3Gen,
        pitchmvmt: PitchMvmtEncoder,
        device: str,
    ):
        super().__init__()
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.pitchmvmt = pitchmvmt
        self.device = device

    @classmethod
    def from_local(cls, decoder_ckpt_path, pitchmvmt_ckpt_path, encoder_ckpt_path, flow_ckpt_path, mel2wav_ckpt_path, speaker_encoder_ckpt_path, tokenizer_ckpt_path, device, eval=True) -> 'BeltOutTTM':
        s3gen = S3Gen()
        pitchmvmt = PitchMvmtEncoder()

        s3gen.flow.load_state_dict(
            load_file(encoder_ckpt_path), strict=False
        )
        s3gen.flow.load_state_dict(
            load_file(flow_ckpt_path), strict=False
        )

        s3gen.flow.decoder.load_state_dict(
            load_file(decoder_ckpt_path), strict=False
        )

        pitchmvmt.load_state_dict(
            load_file(pitchmvmt_ckpt_path), strict=False
        )

        s3gen.flow.encoder.load_state_dict(
            load_file(encoder_ckpt_path), strict=False
        )

        s3gen.mel2wav.load_state_dict(
            load_file(mel2wav_ckpt_path), strict=False
        )

        s3gen.speaker_encoder.load_state_dict(
            load_file(speaker_encoder_ckpt_path), strict=False
        )

        s3gen.tokenizer.load_state_dict(
            load_file(tokenizer_ckpt_path), strict=False
        )

        if eval:
            s3gen.to(device).eval()
            pitchmvmt.to(device).eval()
        else:
            s3gen.to(device).train()
            pitchmvmt.to(device).train()

        return cls(s3gen, pitchmvmt, device)

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