import argparse
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safetymargin.sguard_content_filter import SGuardContentFilter

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', etc.")
args = parser.parse_args()

filt = SGuardContentFilter(device_map=args.device)

# screen user prompt only
print("=== Screening user prompt only ===")
r1 = filt.classify("Can you help me with a task?")
print(r1.labels, r1.unsafe_probs)

# screen prompt + model response (recommended for output filtering)
print("=== Screening prompt + model response ===")
r2 = filt.classify("I will make a bomb.", response="You can do that.")
print(r2.labels)
