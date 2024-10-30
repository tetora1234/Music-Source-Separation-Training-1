import os
import torch
import librosa
import soundfile as sf
import numpy as np
import yaml
from typing import List, Union
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

def get_device(force_cpu: bool = False, device_ids: Union[List[int], int] = 0) -> str:
    """Get appropriate device for processing."""
    if force_cpu:
        return "cpu"
    elif torch.cuda.is_available():
        return f'cuda:{device_ids[0]}' if isinstance(device_ids, list) else f'cuda:{device_ids}'
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def process_audio_file(
    input_file: str,
    output_dir: str,
    model_type: str = 'mdx23c',
    config_path: str = None,
    checkpoint_path: str = '',
    extract_instrumental: bool = False,
    use_tta: bool = False,
    output_format: str = 'wav',
    pcm_type: str = 'PCM_24',
    force_cpu: bool = False,
    device_ids: Union[List[int], int] = 0
) -> List[str]:
    """
    Process an audio file and return paths to the processed output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(force_cpu, device_ids)
    print(f"Using device: {device}")

    # Add current directory to system path for utils import
    current_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    sys.path.append(current_dir)
    from utils import get_model_from_config, demix

    # Load and prepare model
    model, config = get_model_from_config(model_type, config_path)
    
    if checkpoint_path:
        print(f'Loading checkpoint: {checkpoint_path}')
        if model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'state' in state_dict:
                state_dict = state_dict['state']
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    if isinstance(device_ids, list) and len(device_ids) > 1 and not force_cpu:
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    model.eval()
    
    # Load and prepare audio
    print(f"Processing file: {input_file}")
    try:
        mix, sr = librosa.load(input_file, sr=44100, mono=False)
    except Exception as e:
        raise Exception(f'Cannot read track: {input_file}. Error: {str(e)}')
    
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)
    
    mix_orig = mix.copy()
    if hasattr(config, 'inference') and 'normalize' in config.inference and config.inference['normalize'] is True:
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std
    
    # Process audio
    if use_tta:
        print("Using Test Time Augmentation...")
        track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
    else:
        track_proc_list = [mix.copy()]
    
    full_result = []
    for mix_track in track_proc_list:
        waveforms = demix(config, model, mix_track, device, pbar=True, model_type=model_type)
        full_result.append(waveforms)
    
    # Average results
    waveforms = full_result[0]
    for i in range(1, len(full_result)):
        d = full_result[i]
        for el in d:
            if i == 2:
                waveforms[el] += -1.0 * d[el]
            elif i == 1:
                waveforms[el] += d[el][::-1].copy()
            else:
                waveforms[el] += d[el]
    for el in waveforms:
        waveforms[el] = waveforms[el] / len(full_result)
    
    # Get instruments and process
    instruments = config.training.instruments.copy()
    if hasattr(config.training, 'target_instrument') and config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]
    
    if extract_instrumental:
        instr = 'vocals' if 'vocals' in instruments else instruments[0]
        if 'instrumental' not in instruments:
            instruments.append('instrumental')
        waveforms['instrumental'] = mix_orig - waveforms[instr]
    
    # Save output files
    output_files = []
    file_name, _ = os.path.splitext(os.path.basename(input_file))
    
    for instr in instruments:
        estimates = waveforms[instr].T
        if hasattr(config, 'inference') and 'normalize' in config.inference and config.inference['normalize'] is True:
            estimates = estimates * std + mean
        
        if output_format == 'flac':
            output_file = os.path.join(output_dir, f"{file_name}_{instr}.flac")
            subtype = 'PCM_16' if pcm_type == 'PCM_16' else 'PCM_24'
            sf.write(output_file, estimates, sr, subtype=subtype)
        else:
            output_file = os.path.join(output_dir, f"{file_name}_{instr}.wav")
            sf.write(output_file, estimates, sr, subtype='FLOAT')
        
        print(f"Saved: {output_file}")
        output_files.append(output_file)
    
    return output_files

def process_with_model(
    model_name: str,
    input_file: str,
    output_dir: str
) -> str:
    """
    指定されたモデルで音声処理を実行し、出力ファイルのパスを返す。
    """
    yaml_path = rf"C:\Users\user\Downloads\Music-Source-Separation-Training\data\{model_name}\config.yaml"
    
    # YAML設定を読み込む
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 音声処理を実行
    output_files = process_audio_file(
        input_file=input_file,
        output_dir=output_dir,
        model_type=config['model_type'],
        config_path=config['config_path'],
        checkpoint_path=config['start_checkpoint']
    )
    
    # 出力ファイルの最後のファイルを次のモデルの入力に使用
    return output_files[-1]  # 最後の出力ファイルを返す


def main():
    """複数のモデルで音声処理を順次行うメイン関数"""
    # 処理対象の音声ファイル
    input_file = r"C:\Users\user\Downloads\Music-Source-Separation-Training\data\241030_083041.wav"
    output_dir = r"C:\Users\user\Downloads\Music-Source-Separation-Training\data\outputs"
    
    # 使用するモデルのリスト
    model_names = ["HTDemucs4 FT Vocals", "BS Roformer", "MelBand Roformer (anvuew edition)"]
    
    # 各モデルで順次処理
    for model_name in model_names:
        print(f"{model_name}で処理中...")
        input_file = process_with_model(
            model_name=model_name,
            input_file=input_file,
            output_dir=output_dir
        )
    
    # 最終出力ファイルの表示
    return input_file

if __name__ == "__main__":
    input_file = main()
    print("最終出力ファイル:", input_file)
