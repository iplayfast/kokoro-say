import argparse
from kokoro import KModel, KPipeline
import torch
import soundfile as sf
import os

def main():
    parser = argparse.ArgumentParser(description='Kokoro TTS Command Line Interface')
    parser.add_argument('text', help='Text to convert to speech')
    parser.add_argument('--output', '-o', default='output.wav', help='Output audio file path (default: output.wav)')
    parser.add_argument('--voice', '-v', default='af_heart', help='Voice to use (default: af_heart)')
    parser.add_argument('--speed', '-s', type=float, default=1.0, help='Speech speed (default: 1.0)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    args = parser.parse_args()

    # Initialize model
    use_gpu = torch.cuda.is_available() and not args.cpu
    device = 'cuda' if use_gpu else 'cpu'
    model = KModel().to(device).eval()

    # Initialize pipeline
    pipeline = KPipeline(lang_code=args.voice[0], model=False)
    
    # Custom pronunciations (from original code)
    pipeline.g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO' if args.voice[0] == 'a' else 'kˈQkəɹQ'

    # Generate audio
    pack = pipeline.load_voice(args.voice)
    for _, ps, _ in pipeline(args.text, args.voice, args.speed):
        ref_s = pack[len(ps)-1]
        try:
            audio = model(ps, ref_s, args.speed)
            # Save the audio file
            sf.write(args.output, audio.cpu().numpy(), 24000)
            print(f"Audio saved to {args.output}")
            return
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            if use_gpu:
                print("Retrying with CPU...")
                model = model.to('cpu')
                audio = model(ps, ref_s, args.speed)
                sf.write(args.output, audio.cpu().numpy(), 24000)
                print(f"Audio saved to {args.output}")
                return

if __name__ == '__main__':
    main()
