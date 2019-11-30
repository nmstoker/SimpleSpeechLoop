import os
import sys
import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
import colorful as cf
import requests
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """
        Microphone may not support our native processing sampling rate, so
        resample from input_rate to RATE_PROCESS here for webrtcvad and
        deepspeech

        Args:
            data (binary): Input audio stream
            input_rate (int): Input audio rate to resample from
        """
        data16 = np.fromstring(string=data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tostring()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz, blocking if necessary."""
        return self.resample(data=self.buffer_queue.get(),
                             input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def pause(self):
        """Temporarily stop the stream listening."""
        self.stream.stop_stream()
        #self.stream.close()
        #self.pa.terminate()

    def restart(self):
        """Restart the stream listening (when previously paused)."""
        self.stream.start_stream()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.debug("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()


class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=400, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

#def play_wav(wav_file, p):
def play_wav(wav_file):
    # workaround as sound output on my computer is playing up
    # commented out code below may be an alternative (YMMV!)
    os.system(f'aplay {wav_file}')
    # Set chunk size of 1024 samples per data frame
    # chunk = 1024  

    # # Open the sound file 
    # wf = wave.open(wav_file, 'rb')

    # # Open a .Stream object to write the WAV file to
    # # 'output = True' indicates that the sound will be played rather than recorded
    # stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
    #                 #channels = wf.getnchannels(),
    #                 channels = 1,
    #                 #rate = wf.getframerate(),
    #                 rate = 16000,
    #                 output = True,
    #                 output_device_index = 8)
    #                 #output_device_index = 0)

    # # Read data in chunks
    # data = wf.readframes(chunk)

    # # Play the sound by writing the audio data to the stream
    # while len(data) > 0:
    #     stream.write(data)
    #     data = wf.readframes(chunk)

    # # Close and terminate the stream
    # stream.stop_stream()
    # stream.close()

def exit_app():
    logging.warning("Stopping application.")
    echo_line("Stopping application", False)
    sys.exit()

def check_input(input_text, vad_audio):
    vad_audio.pause()
    if input_text.lower() == 'stop':
        exit_app()
    elif input_text.lower().startswith('make a robot noise'):
        play_wav('robot_noise.wav')
    elif input_text.lower().startswith('tell me about'):
        read_wikipedia(input_text)

    elif input_text.lower().startswith('pause'):
        t = 20
        echo_line(f'Pausing for {t} seconds', False)
        time.sleep(20)
        print(cf.bold_coral("Listening (ctrl-C to exit)..."))
    else:
        echo_line(input_text)
    vad_audio.restart()
    return

def read_wikipedia(input_text):
    print_output: print(cf.slateGray("Recognized: {0}".format(cf.bold_white(input_text))))
    input_text = input_text[13:].strip()
    if input_text.strip() != '':
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/{}".format(input_text)
        logging.info('URL: ' + url)
        r = requests.get(url)
        page = r.json()
        if 'extract' in page:
            resp = page["extract"]
            print(resp)
            echo_line(resp, False)
        else:
            echo_line('No details found', False)

def echo_line(input_text, print_output = True):
    filename = 'response.wav'
    if input_text.strip() != '':
        #say_text = 'I heard, ' + input_text
        if len(input_text) <= 2:
            input_text = 'error with text length'
        say_text = input_text
        url = 'http://0.0.0.0:5002/api/tts?text={}'.format(say_text)
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        logging.debug('saved wav for {}'.format(url))
        #p = pyaudio.PyAudio()
        #play_wav(filename, p)
        play_wav(filename)
        #p.terminate()
    if print_output: print(cf.slateGray("Recognized: {0}".format(cf.bold_white(input_text))))
    return


def match_line(input_text):
    # determine closest line to input
    if input_text in ["testing"]:
        line = cf.bold_cornflowerBlue_on_snow(input_text)
    else:
        line = cf.bold_white(input_text)
    return line

def main(ARGS):

    #p = pyaudio.PyAudio()
    play_wav('robot_noise.wav')
    #p.terminate()


    # Load DeepSpeech model
    if os.path.isdir(ARGS.model):
        model_dir = ARGS.model
        ARGS.model = os.path.join(model_dir, 'output_graph.pbmm')
        if not Path(ARGS.model).is_file():
            ARGS.model = os.path.join(model_dir, 'output_graph.pb')
        ARGS.alphabet = os.path.join(model_dir, ARGS.alphabet if ARGS.alphabet else 'alphabet.txt')
        ARGS.lm = os.path.join(model_dir, ARGS.lm)
        ARGS.trie = os.path.join(model_dir, ARGS.trie)

    print(cf.bold_coral('Initializing model...'))
    logging.info("ARGS.model: %s", ARGS.model)
    logging.info("ARGS.alphabet: %s", ARGS.alphabet)
    logging.info("ARGS.beam_width: %s", ARGS.beam_width)
    model = deepspeech.Model(ARGS.model, ARGS.n_features, ARGS.n_context, ARGS.alphabet, ARGS.beam_width)
    if ARGS.lm and ARGS.trie:
        logging.info("ARGS.lm: %s", ARGS.lm)
        logging.info("ARGS.trie: %s", ARGS.trie)
        logging.info("ARGS.lm_alpha: %s", ARGS.lm_alpha)
        logging.info("ARGS.lm_beta: %s", ARGS.lm_beta)
        model.enableDecoderWithLM(ARGS.alphabet, ARGS.lm, ARGS.trie, ARGS.lm_alpha, ARGS.lm_beta)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate)
    print(cf.bold_coral("Listening (ctrl-C to exit)..."))
    frames = vad_audio.vad_collector()

    # Stream from microphone to DeepSpeech using VAD
    spinner = None
    if not ARGS.nospinner: spinner = Halo(spinner='line')
    stream_context = model.setupStream()
    wav_data = bytearray()
    for frame in frames:
        if frame is not None:
            if spinner: spinner.start()
            logging.debug("streaming frame")
            model.feedAudioContent(stream_context, np.frombuffer(frame, np.int16))
            if ARGS.savewav: wav_data.extend(frame)
        else:
            if spinner: spinner.stop()
            logging.debug("end utterence")
            if ARGS.savewav:
                vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                wav_data = bytearray()
            text = model.finishStream(stream_context)
            check_input(text, vad_audio)
            #print(cf.slateGray("Recognized: {0}".format(cf.bold_white(text))))
            stream_context = model.setupStream()

if __name__ == '__main__':
    BEAM_WIDTH = 1000 # changed from 500
    DEFAULT_SAMPLE_RATE = 16000
    LM_ALPHA = 0.75
    LM_BETA = 1.85
    N_FEATURES = 26
    N_CONTEXT = 9

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=2,
        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 2")
    parser.add_argument('--nospinner', action='store_true',
        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
        help="Save .wav files of utterences to given directory")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-a', '--alphabet', default='alphabet.txt',
                        help="Path to the configuration file specifying the alphabet used by the network. Default: alphabet.txt")
    parser.add_argument('-l', '--lm', default='lm.binary',
                        help="Path to the language model binary file. Default: lm.binary")
    parser.add_argument('-t', '--trie', default='trie',
                        help="Path to the language model trie file created with native_client/generate_trie. Default: trie")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device()")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}. Your device may require 44100.")
    parser.add_argument('-nf', '--n_features', type=int, default=N_FEATURES,
                        help=f"Number of MFCC features to use. Default: {N_FEATURES}")
    parser.add_argument('-nc', '--n_context', type=int, default=N_CONTEXT,
                        help=f"Size of the context window used for producing timesteps in the input vector. Default: {N_CONTEXT}")
    parser.add_argument('-la', '--lm_alpha', type=float, default=LM_ALPHA,
                        help=f"The alpha hyperparameter of the CTC decoder. Language Model weight. Default: {LM_ALPHA}")
    parser.add_argument('-lb', '--lm_beta', type=float, default=LM_BETA,
                        help=f"The beta hyperparameter of the CTC decoder. Word insertion bonus. Default: {LM_BETA}")
    parser.add_argument('-bw', '--beam_width', type=int, default=BEAM_WIDTH,
                        help=f"Beam width used in the CTC decoder when building candidate transcriptions. Default: {BEAM_WIDTH}")

    ARGS = parser.parse_args()
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)
    main(ARGS)
