"""TPE-aware dataset processors (parallel copies of LLaMA-Factory's supervised /
unsupervised preprocess functions, extended with tpe path_ids output).

DO NOT import these unconditionally — only when `use_tpe=True`. The base
supervised/unsupervised modules are left untouched.
"""
