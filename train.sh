#!/usr/bin/env bash
CONFIG=$1
GPU=${2:-2}

mim train mmaction $CONFIG --gpus $GPU --launcher pytorch "${@:3}"