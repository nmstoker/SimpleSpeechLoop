# SimpleSpeechLoop

**SimpleSpeechLoop:** a very basic demonstration connecting speech recognition and text-to-speech, using two Mozilla projects:

- [DeepSpeech](https://github.com/mozilla/DeepSpeech)

- [TTS](https://github.com/mozilla/TTS)

## What is this?

It is a bot that listens to what you say with locally running speech recognition and then gives a couple of (limited) responses using text-to-speech

See the demo video here: https://www.youtube.com/watch?v=cDU6Oz1bNoY

**Warning:** it does require that you have working installations of both DeepSpeech and TTS, which may need a certain amount of skill to set up (although that's getting easier and easier thanks to efforts from the devs on the respective projects).

If you run into problems getting either of them set up, the best approach is to carefully read the installation instructions to make sure you haven't missed anything and if you are confident that you've ruled out obvious potential problems then raise it on the relevant Discource forum (giving clear details of what you did - *remember, others will not be able to help you if you are vague on this part*)

- [DeepSpeech Discourse](https://discourse.mozilla.org/c/deep-speech)

- [TTS Discourse](https://discourse.mozilla.org/c/tts)

There are five basic actions:

1. **Echoing:** this is the default - it will echo back whatever the speech recognition thinks it heard you say

2. **"Tell me about ___":** it will look up a Wikipedia document for the word that comes after "Tell me about" and read back the summary.  Good examples are things such as elements, eg "Tell me about iron" returns the summary derived from this page: https://en.wikipedia.org/wiki/Iron

3. **"Make a robot noise":** it will play the file **robot_noise.wav** (*this one can be misheard quite often, at least with my speech models so far!*)

4. **"Pause":** it will pause listening for 20 seconds (so it stops the incessant echoing!!)

5. **"Stop":** it will cause the app to stop running


By looking at the code you should be able to add more. For anything more complicated you'll want a more sophisticated approach beyond this sort of simple loop.

Please note that if there are changes in the APIs of either supporting project as their versions progress, you may need to make adjustments to the code here to get it to keep working.  It should work with version 0.51 of DeepSpeech.  It is effectively an adapted version of the [VAD demo](https://github.com/mozilla/DeepSpeech/tree/master/examples/mic_vad_streaming) from DeepSpeech with TTS bolted on and a few simple tricks to have it say something back to you.

It is shared **"as is"** in the hope that it's helpful in some small way :slightly_smiling_face:

I've only tested it on Linux - best of luck if you try to adapt it for Mac / Windows!

## To run it

0. **Audio Setup:** Make sure you've got a working microphone and audio out plugged into speakers or headphones!

1. **Install both DeepSpeech and TTS** - best to refer to those projects directly.  Recommend you do it in a virtual environment for each (demo.py is run from the DeepSpeech one and the TTS server is run from the TTS one).  You'll need to install demo.py's requirements too (in the DeepSpeech environment) - from memory those are **requests, colorful** and **pyaudio** (but check the file to be sure).

2. **Start the TTS server** - typically you might as well run this locally. Simply make sure that the end-point in demo.py is updated to match (currently set to http://0.0.0.0:5002/api/tts)

3. **Run demo.py** - python demo.py -d 7 -m ../models/your_model_folder/


The parameters are the same as the [VAD demo](https://github.com/mozilla/DeepSpeech/tree/master/examples/mic_vad_streaming) from DeepSpeech.

**-d** is the channel for your microphone (you can check the ALSA channels with **show_alsa_channels.py**)

**-m** is the location of the directory for the DeepSpeech model you plan to use (eg one you've trained / fine-tuned or a pre-trained one)