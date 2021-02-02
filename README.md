# K-Means image clustering (opencv4, c++17)

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
  - [Advanced Usage](#advanced-usage)

## Description

This program learns structure of square pixel groups (res * res) via the k-means clusterisation algorhythm. Here's an example of a working model:

Input             |  Output
:-------------------------:|:-------------------------:
![Morane_wing.png](examples/Morane_wing.png?raw=true "Input")  |  ![Morane_wing_out.png](examples/Morane_wing_out.png?raw=true "Output")

The idea behind this program wa to build multiple clusterisation levels and try to extract soe high-level pattern recognition from it.\
Please note that this program is under heavy **testing & development**. The main pending problems include:\
1) The program should be split into multiple files
2) The algorhythm should be rebuilt to use multiple clusterisation levels

## Installation

To install the app, you need opencv4 preinstalled ([Linux](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html), [MacOS](https://docs.opencv.org/master/d0/db2/tutorial_macos_install.html), [Windows](https://docs.opencv.org/master/d3/d52/tutorial_windows_install.html)).\
Then, you need to build the app.\
First `git clone `, create the build directory in program's folder.\
```cd mkdir build && cd build```

## Usage

To run the app, simply type the following in your Terminal app:
```
$ python3 ~/mp3folderplayer.py ~/path/to/file1 ~/path/to/file2
```
Where the arguments are paths to folders where your music is. This program accepts nested folders.

### Advanced Usage

When the program is started, type h for help. It will provide you the short list with brief description of the main functions:
```
    h -- display help info.

    pr -- previous track. Plays the previous track in the queue.
    [enter]/no input -- play the next track in the queue.

    p -- pause playback and start with the next track. The program doesn't allow to pause music in the middle which provides a continuous music experience.
    s -- silence playback. Silences the music until Enter is pressed.

    sh -- shuffle playlist. Shuffles the queue
    srt -- sort playlist. Sorts the queue

    f -- finish playing. Finish playing and exit the program
```