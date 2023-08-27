
import argparse
import io
import os
import subprocess
import threading
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from tempfile import NamedTemporaryFile
from time import sleep
from typing import List

import numpy as np
import speech_recognition as sr
import torch
import whisper
from scipy.io.wavfile import write as write_wav


class Transcriber():
    def __init__(
        self,
        model: str = "medium",
        microphone_input: int = 56,
        speaker_output: int = 55,
        record_timeout: float = 2,
        phrase_timeout: float = 2,
        wav_output: str = 'output.wav',
        text_output: str = 'output.txt',
    ):
        self.model = model if model == "large" else model + '.en'
        self.microphone_input = microphone_input
        self.speaker_output = speaker_output
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.transcription = ['']
        self.capture_thread = None
        self.wav_output = wav_output
        self.text_output = text_output
        self.data_queue = Queue()
        self._stop = False

    def list_devices(self) -> str:
        command = "pactl list sources short"
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, shell=True)
        return str(result.stdout)

    def audio_capture_thread(self):
        process = None
        ffmpeg_command = [
            "ffmpeg",
            "-f", "pulse",
            "-i", str(self.speaker_output),
            "-f", "pulse",
            "-i", str(self.microphone_input),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "[aout]",
            "-ac", "1",  # Stereo output
            "-ar", "16000",  # Sample rate
            "-f", "wav",  # Output format is WAV
            "-"
        ]
        try:
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

            chunk_size = 16000 * 2
            while True:
                raw_data = process.stdout.read(chunk_size * 2)  # type: ignore
                if not raw_data:
                    break
                audio_samples = np.frombuffer(raw_data, dtype=np.int16)
                self.audio_callback(audio_samples)  # Call the callback function with audio data
        except Exception as ex:
            error = process.stderr.read()  # type: ignore
            print(error)
            print(ex)

    def audio_callback(self, samples):
        self.data_queue.put(samples)

    def stop(self):
        self._stop = True

    def get_current_transcription(self) -> List[str]:
        return self.transcription

    def start(self):
        if os.path.exists(self.wav_output):
            os.remove(self.wav_output)

        self.capture_thread = threading.Thread(target=self.audio_capture_thread)
        self.capture_thread.daemon = True  # Allow the thread to exit when the main program exits
        self.capture_thread.start()

        phrase_time = None
        last_sample = bytes()
        audio_model = whisper.load_model(self.model)

        while not self._stop:
            try:
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not self.data_queue.empty():
                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        if isinstance(data, np.ndarray):
                            data = data.tobytes()
                        last_sample += data

                    # Write wav data to the temporary file as bytes.
                    with open(self.wav_output, 'w+b') as f:
                        f.write(last_sample)

                    # Read the transcription.
                    result = audio_model.transcribe(self.wav_output, fp16=torch.cuda.is_available())
                    text = result['text'].strip()  # type: ignore

                    if phrase_complete:
                        self.transcription.append(text)
                    else:
                        self.transcription[-1] = text

                    # Clear the console to reprint the updated transcription.
                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in self.transcription:
                        print(line)
                    # Flush stdout.
                    print('', end='', flush=True)

                    with open(self.text_output, 'w') as f:
                        f.write('\n'.join(self.transcription))

                    # Infinite loops are bad for processors, must sleep.
                    sleep(0.25)
            except KeyboardInterrupt:
                break

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--speaker_output", default=55,
                        help="Device number for default output device", type=float)
    parser.add_argument("--microphone_input", default=56,
                        help="Device number for the default microphone input device", type=float)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    args = parser.parse_args()

    transcriber = Transcriber(
        model=args.model,
        speaker_output=args.speaker_output,
        microphone_input=args.microphone_input,
        record_timeout=args.record_timeout,
        phrase_timeout=args.phrase_timeout,
    )
    transcriber.start()

if __name__ == "__main__":
    main()
