# Interactive GUI for the Rectangle Packing Problem
Project for the lecture "Optimierungsalgorithmen" at the Technical University Darmstadt.


<img src="https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/screenshot.png" width=800 title="GUI Screenshot">

## Rectangle Packing Problem (RPP)
The _Rectangle Packing Problem (RPP)_ describes the issue of putting ```n``` (almost arbitrary) rectangles into ```N``` squares (boxes) such that ```N``` is minimized.

## Features
### Solving the RPP automatically
Automatic solving using one of two different and efficiently implemented algorithmic approaches (```local search``` and ```greedy search```) with multiple tuning possibilities

<img src="https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/automatic_solve.gif" width=600 title="Automatic RPP Local Search">
<img src="https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/automatic_solve_2.gif" width=600 title="Automatic RPP Greedy Search">


### Solving the RPP by hand

<img src="https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/manual_solve.gif" width=600 title="Manual RPP solving">

### Defining an instance of the RPP
By setting the hyperparameters such as number and sizes of rectangles, box size, etc.
<img src="https://github.com/jonasgrebe/tu-opti-algo-project/blob/main/docs/configuration.gif" width=600 title="Problem configuration">


## How to use
Just execute ```run_gui.py```.

## Implementation
This GUI is implemented using PyGame (for the grid world) and PyGame Menu (for the UI).
