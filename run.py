import langid
import torch
import argparse
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from openvoice import se_extractor

en_ckpt_base = 'checkpoints/base_speakers/EN'
zh_ckpt_base = 'checkpoints/base_speakers/ZH'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
zh_base_speaker_tts = BaseSpeakerTTS(f'{zh_ckpt_base}/config.json', device=device)
zh_base_speaker_tts.load_ckpt(f'{zh_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# load speaker embeddings
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
zh_source_se = torch.load(f'{zh_ckpt_base}/zh_default_se.pth').to(device)

supported_languages = ['zh', 'en']


def predict(prompt, style, audio_file_pth, output_file_pth):
    # initialize an empty info
    text_hint = ''

    # first detect the input language
    language_predicted = langid.classify(prompt)[0].strip()
    print(f"Detected language:{language_predicted}")

    if language_predicted not in supported_languages:
        text_hint += f"[ERROR] The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}\n"
        return (
            text_hint,
            None,
            None,
        )

    if language_predicted == "zh":
        tts_model = zh_base_speaker_tts
        source_se = zh_source_se
        language = 'Chinese'
        if style not in ['default']:
            text_hint += f"[ERROR] The style {style} is not supported for Chinese, which should be in ['default']\n"
            return (
                text_hint,
                None,
                None,
            )
    else:
        tts_model = en_base_speaker_tts
        if style == 'default':
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = 'English'
        if style not in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad',
                         'friendly']:
            text_hint += f"[ERROR] The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']\n"
            return (
                text_hint,
                None,
                None,
            )

    speaker_wav = audio_file_pth

    if len(prompt) < 2:
        text_hint += f"[ERROR] Please give a longer prompt text \n"
        return (
            text_hint,
            None,
            None,
        )
    if len(prompt) > 200:
        text_hint += f"[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        return (
            text_hint,
            None,
            None,
        )

    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
        return (
            text_hint,
            None,
            None,
        )

    src_path = '/tmp/tmp.wav'
    tts_model.tts(prompt, src_path, speaker=style, language=language)

    # Run the tone color converter
    encode_message = "@lwabish"
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_file_pth,
        message=encode_message)

    text_hint += f'''Get response successfully \n'''

    return (
        text_hint,
        output_file_pth,
        speaker_wav,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="", help="text to read")
    parser.add_argument("--audio", type=str, default="", help="audio file to read")
    parser.add_argument("--output", type=str, default="output", help="output directory")
    args = parser.parse_args()
    msg, result, _ = predict(args.text, "default", args.audio, args.output)
    print(msg)
    exit(0 if result is None else 1)
