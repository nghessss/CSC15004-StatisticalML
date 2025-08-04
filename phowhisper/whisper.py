from transformers import pipeline
import subprocess
from pathlib import Path

transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
output = transcriber('./output.mp3')['text']
print(output)


def video_to_mp3(input_file: str, output_file: str):
    """
    Convert a video file (e.g., mp4, mov) to mp3 using ffmpeg.
    Requires ffmpeg.exe in PATH or current directory.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        str(output_path)
    ], check=True)

    return str(output_path)


transcriber = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")
output = transcriber(video_to_mp3('test.mp4', 'output.mp3'))['text']
print(output)