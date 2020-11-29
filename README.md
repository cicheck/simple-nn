# Simple NN

## Table of contents

* [Introduction](#introduction)
* [Usage](#usage)
* [Preview](#preview)


## Introduction

Simple implementation of Neural Network  written purely in python during python course. Does **not** contain  regularization or bias units. To test 3-layers network on sin or quad function run either: 
* **python test.py --type sin --ticks [ticks number]**
* **python test.py --type quad --ticks [ticks number]**

Where **ticks number** corresponds to number of trainings steps before training and animation resets. 

## Usage
1. Make sure your environment meets requirements listed in [requirements.txt](requirements.txt). (pip install -r requirements.txt)
2. To build your own network import class NeuralNetwork from [neural_netowork.py](neural_network.py).
3. To test 3-layers network on sin or quad function run either: **python test.py --type sin --ticks [ticks number]** or **python test.py --type quad --ticks [ticks number]** — **ticks number** corresponds to number of trainings steps before animation resets. 

## Preview

### Sin Function
<p align="center">
  <img src="https://github.com/regin123/simpleNN/blob/master/images/sin.png" alt="drawing" height=350px>
</p>

### Quad Function
<p align="center">
  <img src="https://github.com/regin123/simpleNN/blob/master/images/quad.png" alt="drawing" height=350px>
</p>
