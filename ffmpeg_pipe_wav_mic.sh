#!/usr/bin/env bash

ffmpeg -re -i ${1} -f s16le -ar 48000 -ac 1 - > /tmp/virtualmic
