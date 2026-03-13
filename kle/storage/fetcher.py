#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data fetcher: fetch draw data from remote sources (17500, 917500, ydniu).
"""

import sys
import time
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}
URL_17500 = "https://data.17500.cn/kl8_desc.txt"


def _fetch_17500() -> list:
    data = []
    try:
        r = requests.get(URL_17500, timeout=30, headers=HEADERS)
        r.encoding = "utf-8"
        for line in r.text.strip().split("\n"):
            line = line.strip()
            if len(line) < 10:
                continue
            parts = line.split()
            if len(parts) < 22:
                continue
            period, date_str, numbers = parts[0], parts[1], parts[2:22]
            if not period.isdigit() or len(period) < 6:
                continue
            valid = []
            for n in numbers:
                try:
                    v = int(n)
                    if 1 <= v <= 80:
                        valid.append(v)
                except ValueError:
                    pass
            if len(valid) != 20 or len(set(valid)) != 20:
                continue
            row = {"期数": period, "日期": date_str}
            for i, v in enumerate(valid, 1):
                row[f"红球_{i}"] = v
            data.append(row)
    except Exception as e:
        print(f"17500: {e}", file=sys.stderr)
    return data


def _fetch_917500() -> list:
    data = []
    for path in ("kl81000_cq_asc.txt", "kl81000_asc.txt"):
        url = urljoin("https://data.917500.cn/", path)
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            r.encoding = "utf-8"
            for line in r.text.strip().split("\n"):
                line = line.strip()
                if len(line) < 10:
                    continue
                parts = line.split()
                if len(parts) >= 22:
                    period, numbers = parts[0], parts[2:22]
                elif len(parts) == 21:
                    period, numbers = parts[0], parts[1:21]
                else:
                    continue
                if not period.isdigit() or len(period) < 6:
                    continue
                valid = []
                for n in numbers:
                    if n.isdigit() and 1 <= int(n) <= 80:
                        valid.append(int(n))
                if len(valid) == 20 and len(set(valid)) == 20:
                    row = {"期数": period}
                    for i, v in enumerate(valid, 1):
                        row[f"红球_{i}"] = v
                    data.append(row)
        except Exception as e:
            print(f"917500 {path}: {e}", file=sys.stderr)
    return data


def _fetch_ydniu() -> list:
    data = []
    base = "https://www.ydniu.com/open/"
    for year in range(2020, datetime.now().year + 1):
        page = 1
        while True:
            if page == 1:
                url = urljoin(base, f"kl8History-{year}.html")
            else:
                url = urljoin(base, f"kl8History-{year}/{page}.html")
            try:
                r = requests.get(url, timeout=30, headers=HEADERS)
                r.encoding = "utf-8"
                soup = BeautifulSoup(r.text, "lxml")
                rows_this = 0
                for tr in soup.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 3:
                        continue
                    period = tds[0].get_text(strip=True)
                    nums_txt = tds[2].get_text(strip=True)
                    if not period.isdigit() or len(period) < 6:
                        continue
                    parts = nums_txt.split()
                    valid = []
                    for n in parts:
                        try:
                            v = int(n)
                            if 1 <= v <= 80:
                                valid.append(v)
                        except ValueError:
                            pass
                    if len(valid) == 20 and len(set(valid)) == 20:
                        row = {"期数": period}
                        for i, v in enumerate(valid, 1):
                            row[f"红球_{i}"] = v
                        data.append(row)
                        rows_this += 1
                if rows_this == 0:
                    break
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"ydniu {year} p{page}: {e}", file=sys.stderr)
                break
    return data


class DataFetcher:
    """Fetch from all sources, merge and dedupe by 期数."""

    def fetch_all(self) -> pd.DataFrame:
        all_rows = []
        all_rows.extend(_fetch_917500())
        all_rows.extend(_fetch_ydniu())
        all_rows.extend(_fetch_17500())
        if not all_rows:
            raise RuntimeError("未获取到任何数据")
        by_period = {r["期数"]: r for r in all_rows}
        df = pd.DataFrame(list(by_period.values()))
        return df
