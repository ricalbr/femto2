<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://user-images.githubusercontent.com/45992199/169658724-72260dc7-26c5-4ff4-bdbb-a0ff6635d893.png" alt="femto" width="480">
  </a>
</div>

> Python library for design and fabrication of femtosecond laser written integrated photonic circuits.

## Table of Contents
* [Features](#features)
* [Setup](#setup)
* [Usage](#usage)
* [Issues](#issues)
<!-- * [License](#license) -->

## Features
femto is an open-source package for the design of integrated optical circuits. It allows to define optical components and to compile a .pgm file for the microfabrication of the circuit. The library consists of growing list of parts, which can be composed into larger circuits.
The following components are implemented:
* A waveguide class, allowing easy chaining of bends and straight waveguides
* Directional couplers
* Mach-Zehnder interferometers
* Different types of markers
* G-Code compiler class

<p align="right">(<a href="#top">back to top</a>)</p>


## Setup
Fetmo can be installed via pip via
```bash
git clone git@github.com:ricalbr/femto.git
cd femto
pip install -e .
```

Alternatively, using GitHub Desktop and Spyder IDE, follow

1. Download the repository on GitHub Desktop 
2. Run the following command in the Spyder console
```bash
pip install -e C:\Users\<user>\Documents\GitHub\femto
```
<p align="right">(<a href="#top">back to top</a>)</p>


## Usage
Here a brief example on how to use the library.

First, import all the required packages
```python
from femto import Waveguide, TrenchColumn, Marker, PGMCompiler
import matplotlib.pyplot as plt
```

Define a circuit as a list of waveguides
```python
waveguides = []
```

Start by adding waveguides to the circuit
```python
# SWG
wg = Waveguide()
wg.start([-2, 0, 0.035])
wg.linear([25, 0.0, 0.0], speed=speed)
wg.end()
waveguides.append(wg)

# MZI
for i in range(6):
    wg = Waveguide()
    wg.start([-2, 0.3+i*0.08, 0.035])
    wg.linear([1.48, 0, 0], speed=20)
    wg.arc_mzi((-1)**(i)*0.037, 15, speed=20)
    wg.linear([1.48, 0, 0], speed=20)
    wg.end()
    waveguides.append(wg)
```

Export the G-Code with the following commands
```python
# Waveguide G-Code
with PGMCompiler('MZIs.pgm', ind_rif=1.5/1.33, angle=0.01) as gc:
    gc.rpt(6)
    for wg in waveguides:
        gc.comment(f' +--- Modo: {i+1} ---+')
        gc.point_to_instruction(wg.M)
    gc.endrpt()
    gc.homing()
```
Other example files can be found [here](https://github.com/ricalbr/femto/tree/main/examples)

<p align="right">(<a href="#top">back to top</a>)</p>

## Issues
To request features or report bugs open an issue [here](https://github.com/ricalbr/femto/issues)

<p align="right">(<a href="#top">back to top</a>)</p>
