#!/bin/bash

montage -geometry +0+0 -tile 3x \
        work/ficnn.{linear,circles,moons}/best.png \
        work/picnn.{linear,circles,moons}/best.png \
        tile.png

montage -geometry +0+0 -tile 3x \
        work/ficnn.{linear,circles,moons}/best.pdf \
        work/picnn.{linear,circles,moons}/best.pdf \
        tile.pdf

echo "Created tile.{png,pdf}"
