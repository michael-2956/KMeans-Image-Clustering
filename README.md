# K-Means image clustering (opencv4, c++17)

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
  - [Advanced Usage](#advanced-usage)

## Description

This program learns structure of square pixel groups using the k-means clusterisation algorhythm. Here's an example of a working model:

Input             |  Output
:-------------------------:|:-------------------------:
![Morane_wing.png](examples/Morane_wing.png?raw=true "Input")  |  ![Morane_wing_out.png](examples/Morane_wing_out.png?raw=true "Output")

The idea behind this program wa to build multiple clusterisation levels and try to extract high-level pattern recognition from it. The idea also was to run this from real time input.\
Please note that this program is under heavy **testing & development**. The main pending problems include:
1) The program should be split into multiple files
2) The algorhythm should be rebuilt to use multiple clusterisation levels

## Installation

To install the app, you need opencv4 preinstalled ([Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html), [MacOS](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html), [Windows](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html)).\
\
When it is installed, you need to **build** the app:\
\
First, clone the repo into your current folder and create the build directory in program's folder.\
```$ git clone https://github.com/michael-2956/KMeans-Image-Clustering.git```\
```$ cd KMeans-Image-Clustering && mkdir build && cd build```\
\
Then compile the program. It will check whether opencv is installed correctly for `cmake`.\
```$ cmake .. && cmake --build .```\
\
You can now run the program to check if it works:\
```$ ./KMeans_Image_Clustering```

## Usage

For the examples to work, you need to put images `images/Morane.jpg` and `images/city.png` in the `build` directory. From the `build` directory, run:\
```$ cp ../images/city.png city.png```\
```$ cp ../images/Morane.jpg Morane.jpg```\
\
To train the model and save it to a file, run:\
```$ ./KMeans_Image_Clustering --train city.png```\
When the program asks you whether you want to change settings, type `n`.\
\
To test the resulting model, run:\
```$ ./KMeans_Image_Clustering --test Morane.jpg```

### Advanced Usage

When the program is started, you can modify model settings. Here's their description:
1) `res = 3` -- the resolution of the side of the square the program will run on.
2) `clusters_num = 10` -- number of clusters the program will separate these squares into.
3) `show = true` -- while training, whether to show the resulting cluster canter points.
4) `iterations = 10000` -- number of iterations for training.

## How clusters look like

This is a `res = 3` and `clusters_num = 10` configuration.
![cluster_points.png](examples/cluster_points.png?raw=true "Clusters view")