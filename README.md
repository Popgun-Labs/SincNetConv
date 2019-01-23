## SincConv
A PyTorch 1.0 implementation of the bandpass convolutions described in https://arxiv.org/abs/1811.09725. Compared with normal convolution, this has the following practical benefits for audio-domain models such as speaker or phoneme recognition:
- Fewer parameters
- Faster convergence
- Intepretable filters
- Better performance

Adapted from the official implementation at: https://github.com/mravanelli/SincNet/
Compared to the original implementation, the filter bank construction has been parallelised. Additionally, padding has been added to preserve the length / time dimension of the input audio.

### Authors
- Iqra Shazhad iqra@wearepopgun.com
- Angus Turner angusturner@wearepopgun.com

If you use this code or part of it, please cite the original paper authors!

```
Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
```
