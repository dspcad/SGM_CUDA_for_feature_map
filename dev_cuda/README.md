# CUDA: Semi-Global Stereo Matching for Deep Learning Feature Maps
This work aims to perform semi-global stereo matching onto the feature maps generated from the deep learning neural works.
It consists of three components:
* Matching cost calculation
  * Census transform (hamming distance)
  * SSD/SAD (Euclidian distance)

* Directional cost calculation

  * 8 paths

* Post-processing

  * uniqueness function

I used

# How to compile the code
-----------------------
1. mkdir build
2. cd build
3. cmake -D CMAKE_BUILD_TYPE=Release ..
4. make
5. cd ..


# How to run the example of simple stereo matching
----------------------
An example on how to use the simple stereo matching can be found in
example/demo.cpp. The example loads the left feature maps and right feature maps.
It then uses either **SAD** or **SSD** to compute the matching cost.

- SAD: 0
- SSD: 1

For example:
```
    build/examples/demo left.npy right.npy 0
```


# How to run the example of semi-global stereo matching
----------------------
An example on how to use the semi-global stereo matching can be found in
example/demo_sgsm.cpp. The example loads the left feature maps and right feature maps 
with the following parameters kernel size, number of disparity, P1, and P2.
```
    build/examples/demo_sgsm left.npy right.npy kernel_size num_disparity P1 P2

```

For example:
```
    build/examples/demo_sgsm left.npy right.npy 5 128 150 1500
```

