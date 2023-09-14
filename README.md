# Reverse Intel VPU machine learning model

This repo is for reversing a VPU traffic binary BLOB during ML inference process to recover the full ML model.

The input binary BLOB is captured by sniffing the USB communication between the host and the VPU through Wireshark.

**Run the experiment**
`python main.py [-h] -i <input BIN file name> -o <output file name>`

**Result**
The output contains original '.xml' and '.bin' files which are the representation of the network.

This currently only works for some small CNN networks and need to be improve in the future work.
