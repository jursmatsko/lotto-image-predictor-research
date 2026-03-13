#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
入口：拉取数据。等价于 python main.py get-data
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.get_data import run

if __name__ == "__main__":
    run()
