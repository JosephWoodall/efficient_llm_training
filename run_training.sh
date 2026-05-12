#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
.venv/bin/python3 -u src/training/main_trainer.py > training.log 2>&1
