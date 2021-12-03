#!/usr/bin/env bash
if [ ! -d "log" ]; then
  mkdir log
fi

# python3 module.py | tee log/module.txt
# python3 similar.py | tee log/similar.txt
python3 weighting.py | tee log/weighting.txt

# python3.6 denoise.py -mode torch | tee log/denoise_torch.txt
# python3.6 denoise.py -mode our | tee log/denoise_our.txt
