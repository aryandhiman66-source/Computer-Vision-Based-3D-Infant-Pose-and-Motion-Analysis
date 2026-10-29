"""Quick inspection of the SMIL model parameter dimensions.

Usage (from smil-paper1 root, PowerShell):
  $env:PYTHONPATH = "<repo>/smil"
  python -W "ignore::DeprecationWarning" tools/inspect_model.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'smil'))

from smil_webuser.serialization import load_model

m = load_model(os.path.join(os.path.dirname(__file__), '..', 'smil', 'smil_web.pkl'))

print("=== SMIL model dimensions ===")
print("v_template:", m.v_template.shape)
print("shapedirs :", m.shapedirs.shape)
print("betas     :", m.betas.shape)
print("pose      :", m.pose.shape)
print("trans     :", m.trans.shape)
print("posedirs  :", m.posedirs.shape)
print("weights   :", m.weights.shape)
print("J         :", m.J.shape)
print("kintree   :", m.kintree_table.shape)
print("f (faces) :", m.f.shape)
print()
print("kintree_table (parent rows):")
print(m.kintree_table)
print()
print("num joints =", m.kintree_table.shape[1], "(so pose =", m.kintree_table.shape[1] * 3, "params)")
