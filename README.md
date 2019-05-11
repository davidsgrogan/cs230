convert the original 22050Hz mp3 to an 8000Hz wave:

`ffmpeg -i upperroom_16_ryle_64kb.mp3 -acodec pcm_s16le -ar 8000 upperroom_16_ryle_8khz.wav`

Convert it to a raw binary file consisting solely of 8000 16bit integers per
second:

`ffmpeg -i upperroom_16_ryle_64kb.mp3 -f s16le -acodec pcm_s16le -ar 8000 upperroom_16_ryle_8khz.raw`

The file sizes are the same modulo the wav header that contains info about the
sampling rate and bit depth etc.
