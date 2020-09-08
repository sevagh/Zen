#!/usr/bin/env bash

find libzengarden/ zengarden/ ZenGardenia/ -regex '.*\.\(cu\|h\)' -exec clang-format -style=file -i {} \;
