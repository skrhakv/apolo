#!/bin/bash

bitfusion run -n 1 -p 0.25 -- python3.7 compute-embeddings.py $1 $2
