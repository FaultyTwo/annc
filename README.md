**Notes: This is an old project I wrote when I used to be in the university.**<br>
**This repo is for achiving purpose only.**

## What is this?
ANNC ("Automated Neural Network Chromosomw") is a project for creating neural network models (mainly convolution neural networks) from chromosomes that can be easily understand and quick to use in NAS (Neural Architecture Search) applications.

Chromosomes in the design is aim to be:
- Easy to create
  - With a few understanding of the structure, you can create a PyTorch model using strings or a dict object!
- Flexible
  - Less constraints in model creation. You want a residual connection from the first layer to the last layer? Do you want an inception block with 6 sub paths? Do you want a model with 20 layers? You can do that!

## Requirements:
- torch >= 1.13.1
- automata-lib >= 6.0.2 (v6)

## Recommended Library:
- torchinfo - Turning models into graphs.