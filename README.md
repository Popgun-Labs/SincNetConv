## SincConv
A PyTorch 1.0 implementation of the bandpass convolutions described in https://arxiv.org/abs/1811.09725
Adapted from the official implementation at: https://github.com/mravanelli/SincNet/

Compared to the original implementation, the filter bank construction has been parallelised. 
Additionally, padding has been added to preserve the length / time dimension of the input audio.

### Authors
- Iqra Shazhad iqra@wearepopgun.com
- Angus Turner angusturner@wearepopgun.com
