#!/bin/bash

convert -delay 20 -loop 0 `ls -tr *png` animation.gif
