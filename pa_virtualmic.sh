#!/usr/bin/env bash

pactl load-module module-pipe-source source_name=virtualmic file=/tmp/virtualmic format=s16le rate=48000 channels=1
