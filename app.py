"""Streamlit dashboard — per-portfolio Optuna TPE auto-search.

Run: .venv/Scripts/streamlit.exe run app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import binomtest

from portopt.champions import (
    is_pinned, load_champions, pin_trial, unpin,
    archive_trial_files,
)
from portopt.portfolio import (
    DEFAULT_HOLDOUT_YEARS,
    DEFAULT_MIN_TRAIN_YEARS,
    DEFAULT_TEST_WINDOW_MONTHS,
    WIN_CRITERIA,
    WIN_CRITERIA_PUBLIC,
    Portfolio,
    Scenario,
)

st.set_page_config(
    page_title="Portfolio AI — Diplomovka",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stMetric {background: #F4F6FB; padding: 12px 16px; border-radius: 10px;}
    .stMetric label {font-size: 0.85rem !important; color: #4A5577 !important;}
    .stMetric [data-testid="stMetricValue"] {font-size: 1.55rem !important;}
    .winner-banner {
        background: linear-gradient(90deg, #FFE5B4 0%, #FFF1C4 100%);
        padding: 16px 22px; border-radius: 12px; border-left: 4px solid #E6A100;
        margin: 12px 0;
    }
    section.main > div {max-width: 1500px;}
    .setup-hero {
        background: linear-gradient(135deg, #f4efe4 0%, #f9f4ea 42%, #edf4fb 100%);
        border: 1px solid #d9dfeb;
        border-radius: 22px;
        padding: 22px 24px 18px 24px;
        margin: 8px 0 18px 0;
    }
    .setup-kicker {
        color: #7a6844;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .setup-title {
        color: #1d2d44;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .setup-copy {
        color: #46566f;
        font-size: 1rem;
        line-height: 1.55;
        margin-bottom: 0;
    }
    .setup-hero-compact {
        background: linear-gradient(135deg, #f5f9ff 0%, #edf4ff 48%, #e6f0ff 100%);
        border: 1px solid #d7e4f5;
        border-radius: 22px;
        padding: 22px 24px 18px 24px;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .portfolio-section-marker) {
        background: linear-gradient(135deg, #e6f0ff 0%, #dbe9ff 45%, #d1e2ff 100%);
        border: 1px solid #b8cff0;
        border-radius: 22px;
        padding: 18px 18px 20px 18px;
        margin: 8px 0 4px 0;
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .portfolio-section-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .portfolio-section-marker {
        display: none;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) {
        background: linear-gradient(135deg, #e4f6ea 0%, #d8f0df 45%, #cee9d6 100%);
        border: 1px solid #b9d8c2;
        border-radius: 22px;
        padding: 18px 18px 20px 18px;
        margin: 2px 0 18px 0;
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .run-section-marker {
        display: none;
    }
    .run-hero {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 14px;
    }
    .run-hero-icon {
        font-size: 2.5rem;
        line-height: 1;
    }
    .run-hero-title {
        color: #16324f;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 6px 0;
    }
    .run-hero-copy {
        color: #53647a;
        font-size: 1.03rem;
        line-height: 1.55;
        margin: 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.run-card-marker) {
        background: rgba(255, 255, 255, 0.58);
        border: 1px solid #c8ddd0;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(57, 94, 69, 0.06);
    }
    .run-card-marker {
        display: none;
    }
    .run-top-control-marker {
        display: none;
    }
    .run-card-title {
        color: #18222f;
        font-size: 1.15rem;
        font-weight: 800;
        margin: 0 0 4px 0;
    }
    .run-card-copy {
        color: #607086;
        font-size: 0.98rem;
        line-height: 1.5;
        min-height: 44px;
        margin: 0 0 6px 0;
    }
    .run-card-status {
        color: #5c6f86;
        font-size: 0.9rem;
        line-height: 1.45;
        min-height: 44px;
        margin: 0 0 10px 0;
    }
    .run-slider-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 18px;
        margin-bottom: 8px;
    }
    .run-slider-text {
        flex: 1 1 auto;
        min-width: 0;
    }
    .run-slider-bubble-wrap {
        position: relative;
        height: 32px;
        margin: 2px 0 0 0;
    }
    .run-slider-bubble {
        position: absolute;
        top: 0;
        transform: translateX(-50%);
        background: rgba(255,255,255,0.96);
        border: 1px solid #d9e1ed;
        border-radius: 12px;
        padding: 8px 12px;
        color: #18222f;
        font-size: 0.95rem;
        font-weight: 800;
        line-height: 1;
        box-shadow: 0 8px 18px rgba(80, 101, 133, 0.10);
    }
    .run-slider-bubble::after {
        content: "";
        position: absolute;
        left: 50%;
        bottom: -6px;
        width: 10px;
        height: 10px;
        background: rgba(255,255,255,0.96);
        border-right: 1px solid #d9e1ed;
        border-bottom: 1px solid #d9e1ed;
        transform: translateX(-50%) rotate(45deg);
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) button[kind="primary"] {
        min-height: 56px;
        border-radius: 14px;
        font-size: 1.15rem;
        font-weight: 800;
        box-shadow: 0 10px 24px rgba(43, 110, 221, 0.22);
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) button[kind="secondary"],
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) button[kind="tertiary"],
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) button[kind="secondaryFormSubmit"],
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) button[kind="primaryFormSubmit"] {
        min-height: 52px;
        border-radius: 14px;
        font-size: 1rem;
        font-weight: 700;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle,
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle_locked {
        background: rgba(255,255,255,0.92);
        border: 1px solid #d5dde8;
        border-radius: 14px;
        padding: 10px 14px;
        box-shadow: 0 10px 28px rgba(74, 101, 136, 0.12);
        min-height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle > div,
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle_locked > div {
        width: 100%;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle label,
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle_locked label {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 14px !important;
        width: 100% !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #18222f !important;
        text-align: center !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle [role="switch"],
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) .st-key-run_forever_toggle_locked [role="switch"] {
        transform: scale(1.35);
        margin: 0;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] {
        padding-top: 10px;
        padding-bottom: 4px;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] [data-baseweb="slider"] {
        padding-left: 6px;
        padding-right: 6px;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        height: 8px !important;
        border-radius: 999px !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        width: 28px !important;
        height: 28px !important;
        border: 3px solid #ffffff !important;
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%) !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28) !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:focus,
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 22px rgba(37, 99, 235, 0.34) !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .run-section-marker) div[data-testid="stSlider"] p {
        color: #5f6f86 !important;
        font-size: 0.96rem !important;
        font-weight: 600 !important;
    }
    .run-slider-scale {
        display: grid;
        grid-template-columns: repeat(11, minmax(0, 1fr));
        gap: 0;
        margin-top: 2px;
        color: #627389;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .run-slider-scale span {
        text-align: center;
    }
    .run-slider-scale span:first-child {
        text-align: left;
    }
    .run-slider-scale span:last-child {
        text-align: right;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) {
        align-items: stretch !important;
        gap: 14px !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) > div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
        height: 100%;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) div[data-testid="stVerticalBlockBorderWrapper"]:has(.run-top-control-marker) {
        height: 204px;
        min-height: 204px;
        max-height: 204px;
        width: 100%;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) div[data-testid="stVerticalBlockBorderWrapper"]:has(.run-top-control-marker) > div[data-testid="stVerticalBlock"] {
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) div[data-testid="stElementContainer"]:has(.run-top-control-marker) {
        height: 100%;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) div[data-testid="stElementContainer"]:has(.run-top-control-marker) + div[data-testid="stElementContainer"] {
        flex: 1 1 auto;
    }
    @media (max-width: 980px) {
        div[data-testid="stHorizontalBlock"]:has(.run-top-control-marker) div[data-testid="stVerticalBlockBorderWrapper"]:has(.run-top-control-marker) {
            height: auto;
            min-height: 0;
            max-height: none;
        }
    }
    .run-slider-col-marker,
    .run-toggle-col-marker {
        display: none;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) {
        flex-wrap: wrap !important;
        align-items: stretch !important;
        gap: 14px 14px !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-slider-col-marker) {
        flex: 999 1 520px !important;
        min-width: 320px !important;
        width: auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) {
        flex: 1 1 260px !important;
        min-width: 240px !important;
        width: auto !important;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) .st-key-run_forever_toggle {
        height: 100%;
    }
    div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) .st-key-run_forever_toggle label {
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }
    @media (max-width: 980px) {
        div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-slider-col-marker),
        div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) .st-key-run_forever_toggle {
            min-height: 84px;
            padding: 16px 18px;
        }
        div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) .st-key-run_forever_toggle label {
            justify-content: flex-start !important;
            font-size: 1rem !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.run-toggle-col-marker) > div[data-testid="column"]:has(.run-toggle-col-marker) .st-key-run_forever_toggle [role="switch"] {
            transform: scale(1.85);
        }
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-section-marker) {
        background: linear-gradient(135deg, #fff8e8 0%, #fff0d3 45%, #ffe8be 100%);
        border: 1px solid #ebd29a;
        border-radius: 22px;
        padding: 18px 18px 20px 18px;
        margin: 2px 0 4px 0;
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-section-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-section-marker) .stMetric {
        background: linear-gradient(180deg, #fff4d6 0%, #ffedc3 100%);
        border: 1px solid #efcf8f;
        border-radius: 14px;
        box-shadow: 0 8px 18px rgba(176, 127, 21, 0.08);
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-section-marker) .stMetric label {
        color: #8a6210 !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-section-marker) .stMetric [data-testid="stMetricValue"] {
        color: #5d4208;
    }
    .results-section-marker {
        display: none;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-chart-marker) {
        background: #ffffff;
        border: 1px solid #e6c37a;
        border-radius: 24px;
        padding: 22px 22px 16px 22px;
        margin: 10px 0 12px 0;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-chart-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-chart-marker) .stVegaLiteChart {
        background: #ffffff;
        border-radius: 18px;
        overflow: hidden;
        padding: 8px 10px 2px 10px;
    }
    .results-chart-marker {
        display: none;
    }
    .results-chart-hint {
        background: #ffffff;
        color: #7c4a00;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.5;
        padding: 12px 14px;
        border-radius: 16px;
        border: 1px solid #f1d69c;
        margin: 0 0 14px 0;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-stats-marker) {
        background: linear-gradient(180deg, #ffe7b6 0%, #ffd995 100%);
        border: 1px solid #dfa954;
        border-radius: 18px;
        padding: 14px 14px 10px 14px;
        margin: 8px 0 8px 0;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .results-stats-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .results-stats-marker {
        display: none;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .compare-section-marker) {
        background: linear-gradient(135deg, #f5f5f5 0%, #ebebeb 45%, #dcdcdc 100%);
        border: 1px solid #b5b5b5;
        border-radius: 22px;
        padding: 18px 18px 20px 18px;
        margin: 2px 0 4px 0;
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .compare-section-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .compare-section-marker) .stMetric {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 14px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .compare-section-marker) .stMetric label {
        color: #374151 !important;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .compare-section-marker) .stMetric [data-testid="stMetricValue"] {
        color: #111827;
    }
    .compare-section-marker {
        display: none;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .tech-section-marker) {
        background: linear-gradient(135deg, #f0e4ff 0%, #e4d4ff 45%, #dac7ff 100%);
        border: 1px solid #c0a6e7;
        border-radius: 22px;
        padding: 18px 18px 20px 18px;
        margin: 2px 0 4px 0;
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .tech-section-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .tech-section-marker {
        display: none;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .stats-section-marker) {
        background: linear-gradient(135deg, #ffe9cf 0%, #ffd9ad 45%, #ffc57a 100%);
        border: 1px solid #e1a14e;
        border-radius: 22px;
        padding: 18px 18px 20px 18px;
        margin: 2px 0 4px 0;
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .stats-section-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .stats-section-marker {
        display: none;
    }
    /* Weights chart card — rounded, breathing room on top so the title isn't glued to the edge */
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .weights-card-marker) {
        background: #ffffff;
        border: 1px solid #d8e2f0;
        border-radius: 22px;
        padding: 28px 24px 22px 24px;
        margin: 24px 0 10px 0;
        box-shadow: 0 4px 14px rgba(50, 80, 130, 0.06);
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .weights-card-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .weights-card-marker {
        display: none;
    }
    /* NAV equity overlay card — same rounded white card as weights */
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .nav-card-marker) {
        background: #ffffff;
        border: 1px solid #d8e2f0;
        border-radius: 22px;
        padding: 28px 26px 22px 26px;
        margin: 16px 0 10px 0;
        box-shadow: 0 4px 14px rgba(50, 80, 130, 0.06);
        overflow: hidden;
    }
    div[data-testid="stLayoutWrapper"]:has(> div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] .nav-card-marker) > div[data-testid="stVerticalBlock"] {
        background: transparent;
    }
    .nav-card-marker {
        display: none;
    }
    .nav-ticker-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px 12px;
        margin: 6px 0 16px 0;
        color: #5d6c82;
        font-size: 0.93rem;
    }
    .nav-ticker-strip .t-label {
        color: #16324f;
        font-weight: 600;
        margin-right: 4px;
        white-space: nowrap;
    }
    .nav-ticker-strip .t-codes {
        display: flex;
        flex: 1;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 6px 10px;
    }
    .nav-ticker-strip code {
        background: #eef2f9;
        border: 1px solid #d7e0ee;
        border-radius: 6px;
        padding: 2px 9px;
        font-size: 0.85rem;
        color: #1d2d44;
        font-family: 'JetBrains Mono', 'Consolas', monospace;
    }
    .stats-emphasis {
        font-size: 1.08rem;
        line-height: 1.65;
        color: #5f3a05;
        margin: 6px 0;
    }
    .stats-emphasis strong {
        font-weight: 800;
    }
    .stats-dm-heading {
        font-size: 1.02rem;
        font-weight: 800;
        color: #7a4708;
        margin-bottom: 8px;
    }
    .stats-dm-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 14px;
        margin-top: 8px;
    }
    .stats-dm-panel {
        background: linear-gradient(180deg, #fff3dd 0%, #ffe9c5 100%);
        border: 1px solid #e2b46a;
        border-radius: 16px;
        padding: 10px 12px;
    }
    .stats-dm-heading {
        font-size: 1.02rem;
        font-weight: 800;
        color: #7a4708;
        margin-bottom: 6px;
    }
    .stats-dm-line {
        font-size: 1.04rem;
        line-height: 1.48;
        color: #5f3a05;
        margin: 0 0 8px 0;
    }
    .stats-dm-line:last-child {
        margin-bottom: 0;
    }
    .stats-dm-line strong {
        font-weight: 800;
    }
    .setup-shell {
        background: linear-gradient(135deg, #f5f9ff 0%, #edf4ff 48%, #e6f0ff 100%);
        border: 1px solid #d7e4f5;
        border-radius: 22px;
        padding: 0;
        overflow: hidden;
    }
    .setup-shell summary {
        list-style: none;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        align-items: stretch;
        justify-content: flex-start;
        gap: 10px;
        padding: 22px 24px 18px 24px;
    }
    .setup-shell summary::-webkit-details-marker {
        display: none;
    }
    .setup-shell-summary-text {
        min-width: 0;
        flex: 1 1 auto;
        width: 100%;
        text-align: center;
    }
    .setup-shell-summary-text .setup-kicker,
    .setup-shell-summary-text .setup-title,
    .setup-shell-summary-text .setup-copy {
        width: 100%;
        max-width: 100%;
    }
    .setup-shell-toggle {
        flex: 0 0 auto;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 112px;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #c8cfda;
        background: rgba(255,255,255,0.8);
        color: #22324c;
        font-size: 0.9rem;
        font-weight: 700;
        line-height: 1;
        white-space: nowrap;
        margin-left: 0;
        align-self: center;
        box-shadow: 0 4px 14px rgba(60, 78, 104, 0.08);
    }
    .setup-shell-toggle-close {
        display: none;
    }
    .setup-shell[open] .setup-shell-toggle-open {
        display: none;
    }
    .setup-shell[open] .setup-shell-toggle-close {
        display: inline;
    }
    .setup-shell-content {
        padding: 0 24px 18px 24px;
    }
    .setup-hero-body {
        margin-top: 18px;
        border-top: 1px solid #d9dfeb;
        padding-top: 18px;
    }
    .setup-section-title {
        color: #1d2d44;
        font-size: 1rem;
        font-weight: 800;
        margin: 0 0 8px 0;
    }
    .setup-section-copy {
        color: #5d6c82;
        font-size: 0.93rem;
        line-height: 1.5;
        margin: 0 0 12px 0;
    }
    .setup-meta-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 16px 0;
    }
    .setup-meta-card {
        background: rgba(255,255,255,0.72);
        border: 1px solid #dde5f0;
        border-radius: 16px;
        padding: 14px 15px;
    }
    .setup-meta-label {
        color: #6a7792;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .setup-meta-value {
        color: #16324f;
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .setup-meta-note {
        color: #5d6c82;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .portfolio-pick-card {
        height: 100%;
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
        border: 1px solid #dde5f0;
        border-radius: 18px;
        padding: 14px 14px 12px 14px;
        box-shadow: 0 8px 20px rgba(39, 56, 82, 0.06);
    }
    .portfolio-pick-card.active {
        background: linear-gradient(135deg, #f5f9ff 0%, #edf4ff 48%, #e6f0ff 100%);
        border-color: #d7e4f5;
        box-shadow: 0 10px 26px rgba(54, 74, 104, 0.12);
    }
    .portfolio-pick-title {
        color: #17314e;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }
    .portfolio-pick-copy {
        color: #586980;
        font-size: 0.87rem;
        line-height: 1.45;
        min-height: 58px;
    }
    .portfolio-pick-indicator {
        text-align: center;
        color: #69809d;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    .setup-list-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px 22px;
        margin-top: 10px;
    }
    .setup-list {
        margin: 0;
        padding-left: 1.1rem;
    }
    .setup-list li {
        color: #42536c;
        font-size: 0.91rem;
        line-height: 1.45;
        margin-bottom: 7px;
    }
    .setup-list strong {
        color: #1f3553;
    }
    .method-card {
        background: #f7f9fc;
        border: 1px solid #dde5f0;
        border-radius: 18px;
        padding: 16px 18px;
        min-height: 126px;
    }
    .method-label {
        color: #6a7792;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .method-value {
        color: #16324f;
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .method-note {
        color: #5d6c82;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .disclaimer-inline {
        background: #fff3f3;
        border: 1px solid #efb3b3;
        border-left: 4px solid #c62828;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0 14px 0;
        color: #6f2c2c;
        font-size: 0.98rem;
        line-height: 1.55;
    }
    .disclaimer-inline b {
        font-size: 1.15rem;
        font-weight: 800;
    }
    .gate-shell {
        max-width: 760px;
        margin: 12vh auto 0 auto;
        background: linear-gradient(135deg, #fff1f1 0%, #fff8f8 100%);
        border: 1px solid #efb3b3;
        border-left: 6px solid #c62828;
        border-radius: 22px;
        padding: 24px 26px 18px 26px;
        box-shadow: 0 18px 55px rgba(85, 28, 28, 0.18);
    }
    .gate-kicker {
        color: #a61b1b;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .gate-title {
        color: #7f1d1d;
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 10px;
    }
    .gate-copy {
        color: #6f2c2c;
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0 0 10px 0;
    }
    @media (max-width: 900px) {
        .setup-shell-toggle {
            align-self: stretch;
            width: 100%;
            min-width: 0;
        }
        .setup-meta-grid,
        .setup-list-grid {
            grid-template-columns: 1fr;
        }
    }
    html {
        scroll-behavior: smooth;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent

PRETTY = {
    "ai": "🤖 AI (softmax neurónová sieť)",
    "equal_weight": "⚖️ Equal Weight 1/N",
    "markowitz": "📊 Markowitz",
    "SPY": "📈 SPY (S&P 500)",
}

# Allowed comparison set used across the app and win logic.
SHOWN_BASELINES = {"equal_weight", "markowitz", "black_litterman", "momentum", "SPY"}

PRETTY["black_litterman"] = "Black-Litterman"
PRETTY["momentum"] = "Momentum"

ASSET_NOTES = {
    "AAPL": "Apple pridáva kvalitný technologický rast, silný cash flow a globálnu značku.",
    "MSFT": "Microsoft prináša stabilnejší technologický rast cez cloud, softvér a firemných zákazníkov.",
    "GOOGL": "Alphabet dáva expozíciu na digitálnu reklamu, AI a dominantné internetové služby.",
    "NVDA": "NVIDIA je rastový motor cez čipy pre AI, dátacentrá a vysoko výkonný computing.",
    "META": "Meta pridáva silný reklamný biznis a potenciál rastu cez sociálne platformy a AI.",
    "AMZN": "Amazon spája e-commerce s cloudom, čiže kombinuje rast spotreby aj infraštruktúry.",
    "JPM": "JPMorgan zastupuje silnú banku, ktorá profituje z ekonomickej aktivity a úrokov.",
    "BAC": "Bank of America rozširuje finančný sektor a dáva citlivosť na cyklus ekonomiky.",
    "JNJ": "Johnson & Johnson tlmí volatilitu vďaka defenzívnemu zdravotníctvu a stabilným príjmom.",
    "PFE": "Pfizer pridáva farmaceutickú defenzívu a príjmy menej závislé od ekonomického cyklu.",
    "PG": "Procter & Gamble reprezentuje spotrebný tovar, ktorý ľudia nakupujú aj v slabších časoch.",
    "KO": "Coca-Cola je klasická defenzívna akcia s globálnou značkou a stabilným dopytom.",
    "WMT": "Walmart prináša odolný retail a silnú pozíciu v základnej spotrebe domácností.",
    "XOM": "Exxon Mobil dáva expozíciu na energiu a môže pomáhať pri inflačných alebo komoditných vlnách.",
    "CVX": "Chevron dopĺňa energetický sektor ako ďalší veľký producent s robustným cash flow.",
    "NEM": "Newmont zastupuje zlato, ktoré sa hodí ako poistka pri strese a inflácii.",
    "FCX": "Freeport-McMoRan dáva expozíciu na priemyselné kovy a globálny priemyselný cyklus.",
    "AEM": "Agnico Eagle Mines dopĺňa expozíciu na zlato ako safe-haven časť portfólia.",
    "TLT": "TLT zastupuje dlhé americké dlhopisy, ktoré pomáhajú pri poklese sadzieb a deflácii.",
    "IEF": "IEF pridáva strednodobé štátne dlhopisy ako stabilnejšiu a menej volatilnú dlhopisovú časť.",
    "AGG": "AGG reprezentuje široký dlhopisový trh a zlepšuje celkovú diverzifikáciu portfólia.",
    "LQD": "LQD pridáva kvalitné firemné dlhopisy ako konzervatívny príjmový komponent.",
    "HYG": "HYG dopĺňa výnosnejšie podnikové dlhopisy, ale za cenu vyššieho kreditného rizika.",
}


def _is_shown_baseline(col: str) -> bool:
    """Only allow the supported comparison set."""
    return col in SHOWN_BASELINES


def _asset_note(ticker: str) -> str:
    return ASSET_NOTES.get(ticker, "Toto aktívum dopĺňa diverzifikáciu a špecifický profil daného portfólia.")


# ============================================================
# Helpers
# ============================================================


def read_control(portfolio: Portfolio) -> dict:
    if not portfolio.control_file.exists():
        return {"running": False}
    try:
        return json.loads(portfolio.control_file.read_text(encoding="utf-8"))
    except Exception:
        return {"running": False}


def write_control(portfolio: Portfolio, ctrl: dict) -> None:
    portfolio.ensure_base()
    portfolio.control_file.write_text(
        json.dumps(ctrl, indent=2, default=str), encoding="utf-8"
    )


def start_auto_search(
    portfolio: Portfolio,
    knobs: dict,
    n_trials: int | None,
    seed_hp: dict | None = None,
    seed_label: str | None = None,
) -> tuple[bool, str]:
    portfolio.ensure_base()
    sc = Scenario.from_knobs(portfolio, knobs)
    sc.ensure_dirs()
    ctrl = {
        "running": True,
        "ts": str(pd.Timestamp.now()),
        "portfolio": portfolio.name,
        "scenario_id": sc.id,
        "n_trials": n_trials,
        "n_completed_in_run": 0,
        "knobs": knobs,
        "seed_hp": seed_hp,
        "seed_label": seed_label,
    }
    write_control(portfolio, ctrl)

    script = PROJECT_ROOT / "scripts" / "auto_search_v2.py"
    if not script.exists():
        return False, f"missing {script}"

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    creationflags = 0
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW
    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        log_handle = (portfolio.base_dir / "subprocess.log").open("ab")
        subprocess.Popen(
            [sys.executable, str(script)],
            stdout=log_handle, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT), env=env,
            creationflags=creationflags, close_fds=True,
        )
        return True, "started"
    except Exception as e:
        return False, str(e)


def stop_auto_search(portfolio: Portfolio) -> None:
    ctrl = read_control(portfolio)
    ctrl["running"] = False
    write_control(portfolio, ctrl)


@st.cache_data(ttl=5)
def load_trials(portfolio_name: str, scenario_id_str: str) -> pd.DataFrame:
    p = Portfolio.by_name(portfolio_name)
    path = p.base_dir / "scenarios" / scenario_id_str / "trials.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    # Defensive: race condition in auto_search can write duplicate trial ids when
    # two workers grab the same next-id. Keep only the LATEST row per trial id so
    # downstream `.set_index('trial').loc[t]` returns a scalar, not a Series.
    if "trial" in df.columns and df["trial"].duplicated().any():
        df = df.drop_duplicates(subset="trial", keep="last").reset_index(drop=True)
    return df


@st.cache_data(ttl=30)
def load_baselines(portfolio_name: str, scenario_id_str: str):
    p = Portfolio.by_name(portfolio_name)
    sd = p.base_dir / "scenarios" / scenario_id_str
    if not (sd / "baselines.parquet").exists():
        return None
    metrics = pd.read_parquet(sd / "baselines.parquet")
    returns = pd.read_parquet(sd / "baseline_returns.parquet")
    nav = pd.read_parquet(sd / "baseline_nav.parquet")
    returns.index.name = "date"
    nav.index.name = "date"
    metrics_holdout = None
    if (sd / "baselines_holdout.parquet").exists():
        metrics_holdout = pd.read_parquet(sd / "baselines_holdout.parquet")
    return {
        "metrics": metrics,
        "metrics_holdout": metrics_holdout,
        "returns": returns,
        "nav": nav,
    }


def colored_won_cell(won: bool) -> str:
    return "✅" if won else "❌"


def _format_trial_option(t: int, df: pd.DataFrame) -> str:
    """Robust label for selectbox: handles duplicate trial ids and missing values."""
    try:
        sub = df[df["trial"] == t]
        if len(sub) == 0:
            return f"Pokus #{t}"
        row = sub.iloc[-1]  # most recent if duplicates slipped through
        leader_trial = None
        worst_trial = None
        if "final_nav_eur" in df.columns:
            leader_df = df.dropna(subset=["final_nav_eur"])
            if len(leader_df) > 0:
                try:
                    leader_trial = int(leader_df.loc[leader_df["final_nav_eur"].idxmax(), "trial"])
                except Exception:
                    leader_trial = None
                try:
                    worst_trial = int(leader_df.loc[leader_df["final_nav_eur"].idxmin(), "trial"])
                except Exception:
                    worst_trial = None
        # Výhra/prehra sa hodnotí na HOLDOUTE (TRUE out-of-sample) — to čo Optuna
        # nikdy nevidela. Search-period víťazstvo môže byť overfit.
        won_h_v = (
            row["won_holdout"] if "won_holdout" in row.index
            else row["won_vs_all"] if "won_vs_all" in row.index
            else row["won"] if "won" in row.index
            else False
        )
        won_h_b = bool(won_h_v) if not isinstance(won_h_v, pd.Series) else bool(won_h_v.iloc[-1])
        nav = row["final_nav_eur"] if "final_nav_eur" in row.index else float("nan")
        nav_v = float(nav) if not isinstance(nav, pd.Series) else float(nav.iloc[-1])
        if leader_trial is not None and int(t) == leader_trial:
            _status = "👑 líder"
        elif worst_trial is not None and int(t) == worst_trial:
            _status = "📉 najslabší"
        elif won_h_b:
            _status = "✅ výhra na holdoute"
        else:
            _status = "❌ prehra na holdoute"
        if pd.isna(nav_v):
            return f"Pokus #{int(t)} | {_status}"
        return f"Pokus #{int(t)} | {_status} | {nav_v:,.0f} €".replace(",", " ")
    except Exception:
        return f"Pokus #{t}"


def recompute_won_strict(
    trials_df: pd.DataFrame,
    baselines_data: dict | None,
    criterion: str,
) -> pd.DataFrame:
    """Recompute `won` so AI must beat the allowed comparison set only:
    Markowitz, Black-Litterman, Momentum, and SPY.

    NOTE: For NEW trials (holdout_sharpe column present) `won` is already
    correctly computed in auto_search_v2 from search-period metrics — no
    recomputation needed. This function only fixes LEGACY trials that used
    full-period sharpe vs full-period baselines.
    """
    if trials_df is None or len(trials_df) == 0 or baselines_data is None:
        return trials_df

    # If this is the NEW schema (holdout-aware), `won` is already strict.
    if "holdout_sharpe" in trials_df.columns:
        out = trials_df.copy()
        metrics = baselines_data.get("metrics")
        if metrics is None or len(metrics) == 0:
            if "won_vs_shown" not in out.columns and "won" in out.columns:
                out["won_vs_shown"] = out["won"]
            return out
        shown_idx = [i for i in metrics.index if _is_shown_baseline(str(i))]
        if shown_idx and "sharpe" in metrics.columns:
            max_shown_sharpe = float(metrics.loc[shown_idx, "sharpe"].max())
        else:
            max_shown_sharpe = 0.0
        if shown_idx and "final_nav_eur" in metrics.columns:
            max_shown_nav = float(metrics.loc[shown_idx, "final_nav_eur"].max())
        else:
            max_shown_nav = 0.0
        bench_idx = [i for i in metrics.index if i == "SPY"]
        if bench_idx and "sharpe" in metrics.columns:
            max_bench_sharpe = float(metrics.loc[bench_idx, "sharpe"].max())
        else:
            max_bench_sharpe = 0.0
        if bench_idx and "final_nav_eur" in metrics.columns:
            max_bench_nav = float(metrics.loc[bench_idx, "final_nav_eur"].max())
        else:
            max_bench_nav = 0.0

        if criterion == "total_return" and "final_nav_eur" in out.columns:
            out["won_vs_shown"] = out["final_nav_eur"] > max_shown_nav
            out["won_vs_benchmarks"] = out["final_nav_eur"] > max_bench_nav
        elif criterion == "beat_benchmarks" and "sharpe" in out.columns and "final_nav_eur" in out.columns:
            out["won_vs_shown"] = (out["sharpe"] > max_shown_sharpe) & (
                out["final_nav_eur"] > max_shown_nav
            )
            out["won_vs_benchmarks"] = (out["sharpe"] > max_bench_sharpe) & (
                out["final_nav_eur"] > max_bench_nav
            )
        elif "sharpe" in out.columns:
            out["won_vs_shown"] = out["sharpe"] > max_shown_sharpe
            out["won_vs_benchmarks"] = out["sharpe"] > max_bench_sharpe
        out["won_vs_all"] = out["won_vs_shown"]

        # Holdout-based win check (TRUE OOS). Toto je hlavné kritérium víťazstva
        # z investorského hľadiska: "vyhrala AI na obdobím ktoré Optuna nevidela?"
        metrics_h = baselines_data.get("metrics_holdout")
        if metrics_h is not None and len(metrics_h) > 0:
            shown_h = [i for i in metrics_h.index if _is_shown_baseline(str(i))]
            max_h_nav = (
                float(metrics_h.loc[shown_h, "final_nav_eur"].max())
                if shown_h and "final_nav_eur" in metrics_h.columns else 0.0
            )
            max_h_sharpe = (
                float(metrics_h.loc[shown_h, "sharpe"].max())
                if shown_h and "sharpe" in metrics_h.columns else 0.0
            )
            if criterion == "total_return" and "holdout_final_nav_eur" in out.columns:
                out["won_holdout"] = out["holdout_final_nav_eur"] > max_h_nav
            elif "holdout_sharpe" in out.columns:
                out["won_holdout"] = out["holdout_sharpe"] > max_h_sharpe
            else:
                out["won_holdout"] = out["won_vs_shown"]
            # NaN holdout → fallback na search-based won (no-holdout trials)
            if "holdout_final_nav_eur" in out.columns:
                _mask_no_h = out["holdout_final_nav_eur"].isna()
                out.loc[_mask_no_h, "won_holdout"] = out.loc[_mask_no_h, "won_vs_shown"]
        else:
            out["won_holdout"] = out["won_vs_shown"]

        # `won` reflektuje SKUTOČNÉ víťazstvo (na holdoute = TRUE OOS).
        out["won"] = out["won_holdout"]
        return out

    metrics = baselines_data.get("metrics")
    if metrics is None or len(metrics) == 0:
        return trials_df

    out = trials_df.copy()
    shown_idx = [i for i in metrics.index if _is_shown_baseline(str(i))]
    if shown_idx and "sharpe" in metrics.columns:
        max_shown_sharpe = float(metrics.loc[shown_idx, "sharpe"].max())
    else:
        max_shown_sharpe = 0.0
    if shown_idx and "final_nav_eur" in metrics.columns:
        max_shown_nav = float(metrics.loc[shown_idx, "final_nav_eur"].max())
    else:
        max_shown_nav = 0.0
    max_all_sharpe = max_shown_sharpe
    max_all_nav = max_shown_nav
    # Benchmark-only: SPY
    bench_idx = [i for i in metrics.index if i == "SPY"]
    if bench_idx and "sharpe" in metrics.columns:
        max_bench_sharpe = float(metrics.loc[bench_idx, "sharpe"].max())
    else:
        max_bench_sharpe = 0.0
    if bench_idx and "final_nav_eur" in metrics.columns:
        max_bench_nav = float(metrics.loc[bench_idx, "final_nav_eur"].max())
    else:
        max_bench_nav = 0.0

    if "sharpe" in out.columns:
        out["won_vs_benchmarks"] = out["sharpe"] > max_bench_sharpe
        out["won_vs_all"] = out["sharpe"] > max_all_sharpe
        out["won_vs_shown"] = out["sharpe"] > max_shown_sharpe
    if criterion == "total_return" and "final_nav_eur" in out.columns:
        out["won_vs_benchmarks"] = out["final_nav_eur"] > max_bench_nav
        out["won_vs_all"] = out["final_nav_eur"] > max_all_nav
        out["won_vs_shown"] = out["final_nav_eur"] > max_shown_nav
    elif criterion == "beat_benchmarks" and "sharpe" in out.columns and "final_nav_eur" in out.columns:
        out["won_vs_benchmarks"] = (out["sharpe"] > max_bench_sharpe) & (
            out["final_nav_eur"] > max_bench_nav
        )
        out["won_vs_all"] = (out["sharpe"] > max_all_sharpe) & (
            out["final_nav_eur"] > max_all_nav
        )
        out["won_vs_shown"] = (out["sharpe"] > max_shown_sharpe) & (
            out["final_nav_eur"] > max_shown_nav
        )

    # Legacy schema bez holdoutu — `won_holdout` padá na search-based `won_vs_shown`.
    out["won_holdout"] = out["won_vs_shown"]
    # Replace `won` with the app's supported comparison set.
    out["won"] = out["won_vs_all"]
    return out


def _best_metric_key_for_criterion(criterion: str) -> str:
    return {
        "sharpe": "sharpe",
        "total_return": "final_nav_eur",
        "beat_benchmarks": "objective",
    }.get(criterion, "sharpe")


def _pick_best_trial_row(trials_df: pd.DataFrame, criterion: str) -> pd.Series | None:
    if trials_df is None or len(trials_df) == 0:
        return None
    metric_key = _best_metric_key_for_criterion(criterion)
    if metric_key not in trials_df.columns:
        return None
    usable = trials_df.dropna(subset=[metric_key])
    if len(usable) == 0:
        return None
    return usable.loc[usable[metric_key].idxmax()]


def _safe_float(value) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "—"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def _simple_md_to_html(text: str) -> str:
    safe = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)


def render_learning_curve(
    trials_df: pd.DataFrame,
    best_key: str,
    best_label: str,
    trial_options: list[int],
    current_trial: int | None,
) -> None:
    if len(trials_df) < 5 or best_key not in trials_df.columns:
        st.caption(
            f"📈 Krivka zlepšovania sa zobrazí po **5+ pokusoch** "
            f"({len(trials_df)}/5)."
        )
        return

    st.subheader("📈 Vývoj výsledkov počas pokusov")

    sorted_all = trials_df.sort_values("trial").reset_index(drop=True)
    vals = sorted_all[best_key].astype(float)

    baseline_n = max(3, min(10, len(vals) // 3))
    baseline = float(vals.iloc[:baseline_n].mean())
    denom = max(abs(baseline), 1e-6)
    x_pad = max(2, int(len(sorted_all) * 0.03))

    plot_df = pd.DataFrame({
        "trial": sorted_all["trial"].astype(int),
        "value": vals,
        "pct_vs_start": (vals - baseline) / denom * 100.0,
    })
    plot_df["rolling_q75"] = plot_df["pct_vs_start"].rolling(window=20, min_periods=5).quantile(0.75)
    plot_df["cum_best_pct"] = plot_df["pct_vs_start"].cummax()

    long_df = plot_df.melt(
        id_vars=["trial"],
        value_vars=["pct_vs_start", "rolling_q75", "cum_best_pct"],
        var_name="series",
        value_name="pct",
    ).dropna()
    label_map = {
        "pct_vs_start": "per-pokus",
        "rolling_q75": "rolling top-25 % (q75, 20)",
        "cum_best_pct": "doteraz najlepší",
    }
    long_df["label"] = long_df["series"].map(label_map)

    trial_click = alt.selection_point(
        fields=["trial"], name="trial_pick", on="click", empty=False, clear="dblclick"
    )

    base = alt.Chart(long_df).encode(
        x=alt.X(
            "trial:Q",
            title="Pokus #",
            axis=alt.Axis(format="d"),
            scale=alt.Scale(
                domainMin=max(0, int(plot_df["trial"].min()) - 1),
                domainMax=int(plot_df["trial"].max()) + x_pad,
                nice=False,
            ),
        ),
        y=alt.Y("pct:Q", title="% zmena vs štart"),
        color=alt.Color(
            "label:N",
            legend=alt.Legend(orient="top", title=None),
            scale=alt.Scale(
                domain=["per-pokus", "rolling top-25 % (q75, 20)", "doteraz najlepší"],
                range=["#90A4AE", "#1976D2", "#43A047"],
            ),
        ),
        tooltip=[
            alt.Tooltip("trial:Q", title="pokus #"),
            alt.Tooltip("label:N", title="séria"),
            alt.Tooltip("pct:Q", format="+.1f", title="% zmena"),
        ],
    )
    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#888", strokeDash=[4, 4], opacity=0.6
    ).encode(y="y:Q")
    line_layer = base.transform_filter(alt.datum.label != "per-pokus").mark_line(
        strokeWidth=2.5
    )
    dots_layer = (
        base.transform_filter(alt.datum.label == "per-pokus")
        .mark_circle(stroke="#37474F", strokeWidth=0.5, cursor="pointer")
        .encode(
            size=alt.condition(trial_click, alt.value(220), alt.value(55)),
            opacity=alt.condition(trial_click, alt.value(1.0), alt.value(0.6)),
        )
        .add_params(trial_click)
    )
    chart_pct = (zero_rule + dots_layer + line_layer).properties(height=320)
    st.markdown('<div class="results-chart-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        (
            "<div class='results-chart-hint'>💡 Klikni na sivú bodku a nižšie sa automaticky "
            "prepne detail daného pokusu. Dvojklik zruší výber.</div>"
        ),
        unsafe_allow_html=True,
    )
    chart_event = st.altair_chart(
        chart_pct, use_container_width=True, on_select="rerun"
    )

    if chart_event is not None:
        sel = getattr(chart_event, "selection", None)
        picked = None
        if sel is not None:
            try:
                items = (
                    sel.get("trial_pick")
                    if isinstance(sel, dict)
                    else getattr(sel, "trial_pick", None)
                )
            except Exception:
                items = None
            if items and len(items) > 0:
                try:
                    picked = int(items[0].get("trial"))
                except Exception:
                    picked = None
        last_pick = st.session_state.get("last_chart_pick")
        if picked is None:
            st.session_state.pop("last_chart_pick", None)
        elif picked != last_pick:
            st.session_state["last_chart_pick"] = picked
            if picked in trial_options and picked != current_trial:
                st.session_state["trial_from_chart"] = picked
                st.rerun()

    st.markdown('<div class="results-stats-marker"></div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    best_pct = float(plot_df["pct_vs_start"].max())
    best_idx = int(plot_df["pct_vs_start"].idxmax())
    winner_trial = int(plot_df["trial"].iloc[best_idx])
    rolling_last = (
        float(plot_df["rolling_q75"].dropna().iloc[-1])
        if plot_df["rolling_q75"].notna().any()
        else float("nan")
    )
    col_a.metric(
        "Víťaz nájdený v pokuse",
        f"#{winner_trial}",
        help=(
            f"Pokus, v ktorom AI dosiahla doteraz najlepší výsledok "
            f"({best_label} = {best_pct:+.1f} % vs štart). "
            "Ak je toto číslo blízko poslednému pokusu, AI sa stále zlepšuje; "
            "ak je hlboko v minulosti, ďalšie pokusy už víťaza neprekonali."
        ),
    )
    col_b.metric(
        "Rolling top-25 % vs štart",
        f"{rolling_last:+.1f}%" if not pd.isna(rolling_last) else "—",
        help=(
            "75. percentil z posledných 20 pokusov (min. 5). Ukazuje strop "
            "typických nedávnych výsledkov — ak rastie, AI sa systematicky "
            "zlepšuje a nejde len o jeden šťastný výstrel."
        ),
    )
    col_c.metric(
        "Doteraz najlepší vs štart",
        f"{best_pct:+.1f}%",
        help="O koľko najlepší pokus prekonal úvodný priemer.",
    )

def render_trial_technical_sections(
    chosen_trial: int,
    trial_dir: Path,
    scenario: Scenario,
    knobs: dict,
    wf_df: pd.DataFrame | None,
) -> None:
    st.subheader(f"🔬 Technický rozpad pokusu #{chosen_trial}")
    st.caption(
        "Táto časť je určená na detailný rozbor metodiky: jednotlivé walk-forward okná "
        "a váhy, ktoré AI zvolila."
    )

    if wf_df is not None:
        st.markdown(f"**📊 Walk-forward výsledky (Trial #{chosen_trial})**")
        st.caption(
            "Každý riadok = jedno testovacie okno. AI sa pred každým oknom natrénovala iba na "
            "minulosti, zafixovala váhy a držala ich počas celého okna."
        )
        wf_show = wf_df.copy()
        regimes_path = scenario.base_dir / "window_regimes.parquet"
        if regimes_path.exists():
            try:
                reg_df = pd.read_parquet(regimes_path)
                reg_map = dict(zip(reg_df["window_idx"].astype(int), reg_df["market_regime"]))
                reg_emoji = {
                    "bull": "🟢 bull",
                    "bear": "🔴 bear",
                    "sideways": "🟡 sideways",
                    "unknown": "⚪ ?",
                }
                wf_show["market"] = (
                    wf_show["window_idx"].astype(int).map(reg_map).map(reg_emoji).fillna("⚪ ?")
                )
            except Exception:
                pass

        if "train_start" in wf_show.columns:
            wf_show["train"] = wf_show.apply(
                lambda r: f"{pd.Timestamp(r['train_start']).date()} → {pd.Timestamp(r['train_end']).date()}",
                axis=1,
            )
            wf_show["test"] = wf_show.apply(
                lambda r: f"{pd.Timestamp(r['test_start']).date()} → {pd.Timestamp(r['test_end']).date()}",
                axis=1,
            )
        wf_cols = [
            "window_idx",
            "regime",
            "market",
            "train",
            "test",
            "return",
            "sharpe",
            "max_drawdown",
            "ann_volatility",
            "val_sharpe",
            "n_days",
        ]
        wf_cols = [c for c in wf_cols if c in wf_show.columns]
        wf_fmt = {
            "return": "{:+.1%}",
            "sharpe": "{:.5f}",
            "max_drawdown": "{:.1%}",
            "ann_volatility": "{:.1%}",
            "val_sharpe": "{:.4f}",
        }
        wf_fmt = {k: v for k, v in wf_fmt.items() if k in wf_show.columns}
        wf_styled = (
            wf_show[wf_cols].style.format(wf_fmt)
            .background_gradient(subset=[c for c in ["sharpe", "return"] if c in wf_cols], cmap="Greens")
            .background_gradient(subset=[c for c in ["max_drawdown"] if c in wf_cols], cmap="Reds_r")
        )
        st.dataframe(wf_styled, use_container_width=True, height=min(280, 60 + 35 * len(wf_show)))

    weights_path = trial_dir / "weights.parquet"
    if weights_path.exists():
        weights_df = pd.read_parquet(weights_path)
        if len(weights_df) > 0:
            asset_cols = [c for c in weights_df.columns if c not in ("window_idx", "test_start")]
            long_df = weights_df.melt(
                id_vars=[c for c in ["window_idx", "test_start"] if c in weights_df.columns],
                value_vars=asset_cols,
                var_name="asset",
                value_name="weight",
            )
            long_df = long_df[long_df["weight"].abs() > 0.005]
            _weights_card = st.container()
            with _weights_card:
                st.markdown('<div class="weights-card-marker"></div>', unsafe_allow_html=True)
                st.markdown("**🎯 Váhy zvolené AI na začiatku každého okna**")
                st.caption(
                    "Pre každé okno sa model znovu natrénuje, zvolí nové váhy a tie zostanú "
                    "zafixované na celé ďalšie obdobie."
                )
                if "test_start" in long_df.columns:
                    long_df = long_df.copy()
                    long_df["rok"] = pd.to_datetime(long_df["test_start"]).dt.year.astype(str)
                    x_field = alt.X("rok:O", title="rok", axis=alt.Axis(labelAngle=0))
                else:
                    x_field = alt.X("window_idx:O", title="okno #")
                weights_chart = (
                    alt.Chart(long_df)
                    .mark_bar(size=60)
                    .encode(
                        x=x_field,
                        y=alt.Y("weight:Q", stack="normalize", title="váha", axis=alt.Axis(format="%")),
                        color=alt.Color(
                            "asset:N",
                            scale=alt.Scale(scheme="tableau20"),
                            legend=alt.Legend(orient="right", title="aktívum", columns=1),
                        ),
                        tooltip=[
                            alt.Tooltip("asset:N", title="aktívum"),
                            alt.Tooltip("weight:Q", format=".1%", title="váha"),
                            *([alt.Tooltip("rok:O", title="rok")] if "rok" in long_df.columns else []),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(weights_chart, use_container_width=True)


# ============================================================
# History / Experiments
# ============================================================
# Each portfolio has portfolios/<name>/history/<exp_id>/ containing a saved run.
# Layout matches a scenario folder (trials.parquet, trial_data/, baselines.*).
# Plus a metadata.json with name, notes, summary stats, timestamp.


import re as _re


def _safe_experiment_id(name: str) -> str:
    """Sanitize user-input name → safe filesystem id, suffixed by timestamp."""
    base = _re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()) or "experiment"
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}__{ts}"


def _save_scenario_as_experiment(
    sc: Scenario, name: str, notes: str = ""
) -> tuple[Path, dict]:
    """Copy scenario data to portfolios/<portfolio>/history/<exp_id>/.

    Active scenario is NOT cleared here — caller decides whether to wipe.
    Returns (exp_dir, metadata).
    """
    import shutil as _sh
    exp_id = _safe_experiment_id(name)
    history_root = sc.portfolio.base_dir / "history" / exp_id
    history_root.mkdir(parents=True, exist_ok=True)
    # Copy known artifacts (best-effort).
    items_files = [
        "trials.parquet",
        "baselines.parquet",
        "baselines_holdout.parquet",
        "baseline_returns.parquet",
        "baseline_nav.parquet",
        "baselines_walkforward.parquet",
        "window_regimes.parquet",
        "best_predictor.pt",
        "knobs.json",
        "optuna.db",
    ]
    moved: list[str] = []
    for fname in items_files:
        src = sc.base_dir / fname
        if src.exists():
            try:
                _sh.copy2(str(src), str(history_root / fname))
                moved.append(fname)
            except Exception:
                pass
    # Copy trial_data/ (per-trial NN + parquets)
    src_td = sc.base_dir / "trial_data"
    if src_td.exists():
        try:
            _sh.copytree(str(src_td), str(history_root / "trial_data"))
            moved.append("trial_data/")
        except Exception:
            pass
    # Compute summary stats
    summary = {}
    try:
        df = pd.read_parquet(sc.trials_db) if sc.trials_db.exists() else pd.DataFrame()
        if len(df) > 0:
            summary["n_trials"] = int(len(df))
            for k in ("sharpe", "holdout_sharpe", "final_nav_eur", "holdout_final_nav_eur"):
                if k in df.columns:
                    summary[f"best_{k}"] = float(df[k].max())
            if "won" in df.columns:
                summary["wins"] = int(df["won"].sum())
    except Exception:
        pass
    meta = {
        "id": exp_id,
        "name": name.strip(),
        "notes": notes.strip(),
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "portfolio": sc.portfolio.name,
        "scenario_id": sc.id,
        "knobs": sc.knobs,
        "moved_files": moved,
        "summary": summary,
    }
    (history_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return history_root, meta


def _wipe_active_scenario(sc: Scenario) -> None:
    """Delete the active scenario folder so the portfolio appears 'clean'."""
    import shutil as _sh
    if sc.base_dir.exists():
        try:
            _sh.rmtree(str(sc.base_dir))
        except Exception:
            # Files may be locked (Streamlit/optuna). Best-effort only.
            pass


def _list_experiments(p: Portfolio) -> list[tuple[Path, dict]]:
    """Return saved experiments for this portfolio, newest first."""
    hroot = p.base_dir / "history"
    if not hroot.exists():
        return []
    out: list[tuple[Path, dict]] = []
    for d in sorted(hroot.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        mp = d / "metadata.json"
        meta: dict = {}
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
            except Exception:
                pass
        out.append((d, meta))
    return out


def _list_active_scenarios(p: Portfolio) -> list[Path]:
    """Return scenario dirs for this portfolio that still contain active trial data."""
    sroot = p.base_dir / "scenarios"
    if not sroot.exists():
        return []
    out: list[Path] = []
    for d in sorted(sroot.iterdir()):
        if not d.is_dir():
            continue
        if (
            (d / "trials.parquet").exists()
            or (d / "trial_data").exists()
            or (d / "optuna.db").exists()
        ):
            out.append(d)
    return out


def _count_trials_in_active_scenarios(p: Portfolio) -> int:
    total = 0
    for sdir in _list_active_scenarios(p):
        tp = sdir / "trials.parquet"
        if not tp.exists():
            continue
        try:
            total += len(pd.read_parquet(tp))
        except Exception:
            pass
    return total


def _wipe_portfolio_active_scenarios(p: Portfolio) -> None:
    import shutil as _sh
    sroot = p.base_dir / "scenarios"
    if sroot.exists():
        try:
            _sh.rmtree(str(sroot))
        except Exception:
            pass


def _portfolio_archives_dir(p: Portfolio) -> Path:
    return p.base_dir / "scenario_archives"


def _archive_portfolio_active_scenarios(p: Portfolio) -> tuple[Path, dict]:
    import shutil as _sh

    sroot = p.base_dir / "scenarios"
    if not sroot.exists():
        raise FileNotFoundError("Pre toto portfólio nie sú žiadne aktívne scenáre.")

    ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_root = _portfolio_archives_dir(p) / ts
    archive_root.mkdir(parents=True, exist_ok=True)

    scenario_count = len(_list_active_scenarios(p))
    trial_count = _count_trials_in_active_scenarios(p)
    _sh.move(str(sroot), str(archive_root / "scenarios"))
    (p.base_dir / "scenarios").mkdir(parents=True, exist_ok=True)

    meta = {
        "archived_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "portfolio": p.name,
        "scenario_count": scenario_count,
        "trial_count": trial_count,
    }
    (archive_root / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return archive_root, meta




# ============================================================
# Main UI
# ============================================================

if "disclaimer_accepted" not in st.session_state:
    st.session_state["disclaimer_accepted"] = False

if not st.session_state["disclaimer_accepted"]:
    st.markdown(
        """
        <div class="gate-shell">
            <div class="gate-kicker">Dôležité upozornenie</div>
            <div class="gate-title">Najprv si prosím potvrď obmedzenia tejto aplikácie</div>
            <p class="gate-copy">
                Toto je <b>teoretická a akademická práca</b>. Výstupy aplikácie slúžia na demonštráciu metodiky,
                nie ako investičné odporúčania, návod na obchodovanie ani garancia budúcich výnosov.
            </p>
            <p class="gate-copy">
                Model má obmedzenia, pracuje s historickými dátami a jeho výsledky sa v reálnom svete nemusia zopakovať.
                Aplikáciu preto treba chápať ako experimentálny analytický nástroj, nie ako hotového investičného poradcu.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _gate_left, _gate_mid, _gate_right = st.columns([1.4, 1.2, 1.4])
    with _gate_mid:
        if st.button("Rozumiem a pokračovať", type="primary", use_container_width=True):
            st.session_state["disclaimer_accepted"] = True
            st.rerun()
    st.stop()

components.html(
    """
    <script>
    const STYLE_ID = "codex-toolbar-nav-style";
    const HOST_ID = "codex-toolbar-nav";
    const styleText = `
    #${HOST_ID} {
        position: absolute;
        left: 1rem;
        right: 7.5rem;
        top: 0;
        bottom: 0;
        display: grid;
        grid-template-columns: minmax(0, 1fr) 5.5rem;
        align-items: center;
        column-gap: 0;
        color: #7f1d1d;
        font-family: "Source Sans Pro", sans-serif;
        pointer-events: none;
    }
    #${HOST_ID} .codex-toolbar-links {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        align-items: center;
        justify-content: stretch;
        gap: 0;
        justify-self: stretch;
        width: 100%;
        height: 100%;
        pointer-events: auto;
    }
    #${HOST_ID} .codex-toolbar-spacer {
        width: 100%;
        height: 1px;
    }
    #${HOST_ID} .codex-toolbar-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        min-width: 0;
        height: 100%;
        padding: 0.2rem 0.4rem;
        border-radius: 0;
        border-top: 1px solid #d8dee8;
        border-bottom: 1px solid #d8dee8;
        border-right: 1px solid #d8dee8;
        border-left: 0;
        color: #24324a;
        text-decoration: none;
        font-size: 0.72rem;
        font-weight: 700;
        line-height: 1.05;
        white-space: nowrap;
        box-sizing: border-box;
    }
    #${HOST_ID} .codex-toolbar-pill:first-child {
        border-left: 1px solid #d8dee8;
        border-top-left-radius: 8px;
        border-bottom-left-radius: 8px;
    }
    #${HOST_ID} .codex-toolbar-pill:last-child {
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    #${HOST_ID} .codex-toolbar-pill:nth-child(1) {
        background: #eaf2ff;
        border-color: #bfd2ff;
        color: #1d4ed8;
    }
    #${HOST_ID} .codex-toolbar-pill:nth-child(2) {
        background: #edfbee;
        border-color: #b9e6be;
        color: #15803d;
    }
    #${HOST_ID} .codex-toolbar-pill:nth-child(3) {
        background: #fff6df;
        border-color: #f4d79a;
        color: #b45309;
    }
    #${HOST_ID} .codex-toolbar-pill:nth-child(4) {
        background: #fff0f6;
        border-color: #f1bfd6;
        color: #be185d;
    }
    #${HOST_ID} .codex-toolbar-pill:nth-child(5) {
        background: #f2efff;
        border-color: #cbc2ff;
        color: #6d28d9;
    }
    #${HOST_ID} .codex-toolbar-pill:nth-child(6) {
        background: #ecfbff;
        border-color: #b8e7f2;
        color: #0f766e;
    }
    #${HOST_ID} .codex-toolbar-pill:hover {
        filter: brightness(0.97);
    }
    div[data-testid="stToolbar"] {
        position: relative;
    }
    @media (max-width: 1200px) {
        #${HOST_ID} {
            grid-template-columns: minmax(0, 1fr) 5.5rem;
        }
    }
    @media (max-width: 980px) {
        #${HOST_ID} {
            left: 0.5rem;
            right: 5.8rem;
            grid-template-columns: minmax(0, 1fr) 4.6rem;
        }
        #${HOST_ID} .codex-toolbar-links {
            gap: 6px;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            justify-self: stretch;
        }
        #${HOST_ID} .codex-toolbar-pill {
            padding: 0.26rem 0.28rem;
            font-size: 0.68rem;
        }
    }
    `;
    const navHtml = `
        <div class="codex-toolbar-links">
            <a class="codex-toolbar-pill" href="#sekcia-nastavenie">Nastavenie</a>
            <a class="codex-toolbar-pill" href="#sekcia-spustenie">Spustenie</a>
            <a class="codex-toolbar-pill" href="#sekcia-prehlad">Výsledky</a>
            <a class="codex-toolbar-pill" href="#sekcia-porovnanie">Porovnanie</a>
            <a class="codex-toolbar-pill" href="#sekcia-technicke">Technické</a>
            <a class="codex-toolbar-pill" href="#sekcia-statistika">Štatistika</a>
        </div>
        <div class="codex-toolbar-spacer"></div>
    `;

    function mountToolbarNav() {
        const doc = window.parent.document;
        const toolbar = doc.querySelector('div[data-testid="stToolbar"]');
        if (!toolbar) return false;

        let styleEl = doc.getElementById(STYLE_ID);
        if (!styleEl) {
            styleEl = doc.createElement("style");
            styleEl.id = STYLE_ID;
            styleEl.textContent = styleText;
            doc.head.appendChild(styleEl);
        }

        let host = doc.getElementById(HOST_ID);
        if (!host) {
            host = doc.createElement("div");
            host.id = HOST_ID;
            toolbar.appendChild(host);
        }
        host.innerHTML = navHtml;

        host.querySelectorAll('a[href^="#"]').forEach((link) => {
            link.onclick = (event) => {
                event.preventDefault();
                const id = link.getAttribute("href").slice(1);
                const target = doc.getElementById(id);
                if (target) {
                    target.scrollIntoView({ behavior: "smooth", block: "start" });
                }
                try {
                    window.parent.location.hash = id;
                } catch (e) {}
            };
        });
        return true;
    }

    let attempts = 0;
    const timer = setInterval(() => {
        attempts += 1;
        if (mountToolbarNav() || attempts > 80) {
            clearInterval(timer);
        }
    }, 250);
    </script>
    """,
    height=0,
)
st.markdown("# 🎓 Optimalizácia portfólia pomocou AI")
st.markdown(
    """
    <div class="disclaimer-inline">
        <b>Upozornenie:</b> teoretická a akademická práca, nie investičné odporúčanie.
        Model pracuje s historickými dátami a výsledky sa v reálnom svete nemusia zopakovať.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div id="sekcia-vysvetlivky"></div>', unsafe_allow_html=True)
with st.expander("ℹ️ Vysvetlivky — ako funguje AI model", expanded=False):
    st.markdown(
        """
**Cieľ.** AI sa pozerá na historické denné výnosy aktív a navrhuje, **akú časť portfólia má držať každé aktívum** (tzv. *váhy*) tak, aby maximalizovala dlhodobý výnos pri zachovaní diverzifikácie.

### 🧠 Architektúra modelu
- **Softmax neurónová sieť (MLP)** — viacvrstvový perceptron, ktorý na výstupe používa funkciu *softmax* — tá zaručuje, že **súčet váh = 100 %** a každá váha je medzi 0 a max. limit (typicky 20 % na aktívum).
- **Vstup (features):** posledných **60 obchodných dní** výnosov všetkých aktív v portfóliu + **VIX index strachu** (3 indikátory sentimentu — z-score, 21-dňová zmena, level/40).
- **Výstup:** vektor váh dĺžky N, kde N = počet aktív v portfóliu (napr. Aggressive = 14 aktív).

### 📅 Walk-forward tréning
Model sa **netrénuje raz, ale opakovane** — pre každé 12-mesačné okno:
1. **Tréning** na všetkých dátach od začiatku obdobia po začiatok okna
2. **Zmrazenie váh** — model navrhne váhy a tie sa **držia celý rok** (bez intra-year rebalancingu, kvôli daňovej efektivite)
3. **Hodnotenie** výsledkov na nasledujúcich 12 mesiacoch
4. **Posun ďalej** — nové okno, viac dát, nový tréning

Tým sa simuluje reálne investovanie: model nevidí budúcnosť, len minulosť.

### 🔒 TRUE Holdout (out-of-sample)
- **Posledné 3 roky** sú **úplne uzamknuté** pred Optunou počas hľadania hyperparametrov.
- Optuna optimalizuje len na **search-perióde** (~7 rokov).
- Po skončení sa model **raz** vyhodnotí na holdoute → **najpoctivejší odhad reálnej výkonnosti**.
- Ak Sharpe na holdoute výrazne klesne (> 50 %), model je **pretrénovaný** (overfit).

### 🎯 Hyperparameter search (Optuna)
**Optuna TPE** (Tree-structured Parzen Estimator) automaticky vyberá kombinácie hyperparametrov (počet vrstiev, šírka, learning rate, dropout, weight decay, batch size, epochs...). Každý pokus = jeden tréning sieť na všetkých walk-forward oknách + vyhodnotenie.

### ⚖️ Porovnávacie stratégie (baseline-y)
AI sa porovnáva s **5 etablovanými prístupmi**:

| Stratégia | Logika |
|---|---|
| **Equal Weight 1/N** | rovnaké váhy pre všetky aktíva (najjednoduchší benchmark) |
| **Markowitz** | klasická mean-variance optimalizácia (1952) s Ledoit-Wolf shrinkage |
| **Black-Litterman** | Bayesian rozšírenie Markowitza, kombinuje rovnováhu trhu s názormi |
| **Momentum** | preferuje aktíva, ktoré v poslednom čase rástli (12-mesačný lookback) |
| **SPY (S&P 500)** | pasívna investícia do amerického trhu (čistý market benchmark) |

### 📊 Štatistická validácia
- **Sharpe Bootstrap CI** (Politis-Romano stationary bootstrap) — 95 % interval spoľahlivosti pre Sharpe ratio
- **PSR** (Probabilistic Sharpe Ratio, Bailey-Lopez de Prado 2012) — pravdepodobnosť, že skutočný Sharpe > 0
- **DSR** (Deflated Sharpe Ratio, Bailey-Lopez de Prado 2014) — koriguje na multiple-testing (Optuna skúsila stovky kombinácií, jedna výhra môže byť šťastím)
- **Diebold-Mariano** — test, či sú rozdiely v denných výnosoch medzi AI a baseline štatisticky významné
- **PBO** (Probability of Backtest Overfitting, CSCV) — pravdepodobnosť, že vybraný „víťaz" je len artefakt hľadania

### 🏆 Kritérium víťazstva
Pokus „**vyhráva na holdoute**" iba ak na posledných 3 rokoch (ktoré Optuna nikdy nevidela) **prekoná najlepší baseline** v zvolenej metrike (typicky konečné NAV alebo Sharpe). Search-period výhry sa rátajú samostatne — môžu byť overfit.
        """
    )

st.markdown('<div id="sekcia-nastavenie"></div>', unsafe_allow_html=True)

# ----- Portfolio loading -----
portfolios = Portfolio.list_all()
if not portfolios:
    st.error("Žiadne portfólia v `portfolios/`. Opraviť konfiguráciu.")
    st.stop()

_portfolio_names = [p.name for p in portfolios]
_chosen_name = st.session_state.get("chosen_portfolio_name") or _portfolio_names[0]
try:
    _default_portfolio_idx = _portfolio_names.index(_chosen_name)
except ValueError:
    _default_portfolio_idx = 0
choice_name = _portfolio_names[_default_portfolio_idx]
picker_block = st.container(border=True)
with picker_block:
    st.markdown('<div class="portfolio-section-marker"></div>', unsafe_allow_html=True)
    st.markdown("## Výber portfólia")
    st.caption(
        "Tu si vyberieš investičný profil a hneď vidíš, aké aktíva bude mať AI k dispozícii."
    )
    _pick_cols = st.columns(len(portfolios), gap="small")
    for _col, _portfolio_option in zip(_pick_cols, portfolios):
        _is_active = _portfolio_option.name == choice_name
        with _col:
            st.markdown(
                f"""
                <div class="portfolio-pick-card {'active' if _is_active else ''}">
                    <div class="portfolio-pick-title">{_portfolio_option.display_name}</div>
                    <div class="portfolio-pick-copy">{_portfolio_option.description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _left, _mid, _right = st.columns([2.2, 1, 2.2])
            with _mid:
                if st.button(
                    "◉" if _is_active else "○",
                    key=f"portfolio_picker_{_portfolio_option.name}",
                    use_container_width=True,
                    type="primary" if _is_active else "secondary",
                ):
                    st.session_state["chosen_portfolio_name"] = _portfolio_option.name
                    st.rerun()
st.session_state["chosen_portfolio_name"] = choice_name
portfolio = next(
    (p for p in portfolios if p.name == choice_name),
    portfolios[0],
)

_ctrl = read_control(portfolio)
is_running = bool(_ctrl.get("running", False))
ctrl = _ctrl

# ----- Knobs -----
@st.cache_data(ttl=600)
def _data_coverage(tickers_tuple: tuple[str, ...]) -> dict:
    """Return the date range we actually have for this portfolio's tickers.

    Locks the date pickers to the intersection of all required tickers' coverage,
    so users can't pick a start before the data exists or an end in the future
    (both produce silent failures or empty backtests downstream).
    """
    from portopt.utils.io import DATA_PROCESSED
    p = DATA_PROCESSED / "log_returns.parquet"
    if not p.exists():
        return {"start": None, "end": None, "missing_tickers": list(tickers_tuple)}
    lr = pd.read_parquet(p)
    available = [t for t in tickers_tuple if t in lr.columns]
    missing = [t for t in tickers_tuple if t not in lr.columns]
    if not available:
        return {"start": None, "end": None, "missing_tickers": missing}
    sub = lr[available].dropna(how="any")
    if sub.empty:
        return {"start": None, "end": None, "missing_tickers": missing}
    return {
        "start": sub.index[0].date(),
        "end": sub.index[-1].date(),
        "missing_tickers": missing,
    }


_cov = _data_coverage(tuple(portfolio.tickers))
_data_start = _cov["start"]
_data_end = _cov["end"]
_missing_tickers = _cov["missing_tickers"]

import datetime as _dt
d = portfolio.defaults
date_start = _dt.date(2013, 1, 1)
date_end = _dt.date(2022, 12, 31)
if _data_start is not None and (date_start < _data_start or date_end > _data_end):
    st.error(
        f"❌ Fixovaný rozsah 2013-01-01 → 2022-12-31 nie je celý v dátach "
        f"({_data_start} → {_data_end}). Stiahni chýbajúce dáta."
    )
    st.stop()
if _data_start is None:
    st.error(
        f"❌ Pre tickery `{portfolio.tickers}` nemáme žiadne spoločné dáta. "
        f"Chýbajúce: {_missing_tickers}"
    )
    st.stop()
# Fixed defaults used by steps 2-4 of the guide (these are NOT user choices —
# they are pre-configured methodology baked into the project).
FIXED_HOLDOUT_YEARS = DEFAULT_HOLDOUT_YEARS          # 3 years
FIXED_TEST_WINDOW_MONTHS = DEFAULT_TEST_WINDOW_MONTHS  # 12 = annual (tax-efficient: >1y holding)
FIXED_MAX_WEIGHT = 0.20                               # hard cap 20% per asset
FIXED_MONTHLY_DEPOSIT = float(d.monthly_deposit_eur)

win_criterion = d.win_criterion if d.win_criterion in WIN_CRITERIA_PUBLIC else "sharpe"
holdout_years = FIXED_HOLDOUT_YEARS
test_window_months = FIXED_TEST_WINDOW_MONTHS
max_weight = FIXED_MAX_WEIGHT
monthly_deposit = FIXED_MONTHLY_DEPOSIT
_holdout_start = date_end - _dt.timedelta(days=365 * FIXED_HOLDOUT_YEARS)
_available_tickers = len(portfolio.tickers) - len(_missing_tickers)
_goal_label = {
    "sharpe": "Najvyšší Sharpe pri zvolenom profile",
    "total_return": "Maximálny dlhodobý rast hodnoty",
    "beat_benchmarks": "Prekonať benchmarky v riziku aj výnose",
}.get(win_criterion, "Maximálny dlhodobý rast hodnoty")

_asset_lines = [
    f"<li><strong>{ticker}:</strong> {_asset_note(ticker)}</li>"
    for ticker in portfolio.tickers
]
_asset_mid = (len(_asset_lines) + 1) // 2
_asset_list_left = "".join(_asset_lines[:_asset_mid])
_asset_list_right = "".join(_asset_lines[_asset_mid:])

with picker_block:
    selected_portfolio_block = st.container(border=True)
    with selected_portfolio_block:
        st.markdown(
            f"""
            <details class="setup-shell">
                <summary>
                    <div class="setup-shell-summary-text">
                        <div class="setup-kicker">Vybrané portfólio</div>
                        <div class="setup-title">{portfolio.display_name}</div>
                        <p class="setup-copy">{portfolio.description}</p>
                    </div>
                    <div class="setup-shell-toggle">
                        <span class="setup-shell-toggle-open">Bližšie informácie</span>
                        <span class="setup-shell-toggle-close">Skryť</span>
                    </div>
                </summary>
                <div class="setup-shell-content">
                    <div class="setup-hero-body" style="margin-top:0;">
                        <div class="setup-section-title">Investičný profil</div>
                        <p class="setup-section-copy">
                            Portfólio určuje, aké aktíva má AI k dispozícii. Rizikový štýl teda vychádza priamo
                            z výberu portfólia, nie z dodatočného dotazníka.
                        </p>
                        <div class="setup-meta-grid">
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Aktíva v tomto profile</div>
                                <div class="setup-meta-value">{_available_tickers}/{len(portfolio.tickers)}</div>
                                <div class="setup-meta-note">Počet aktív, z ktorých môže AI skladať portfólio.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Dostupné dáta</div>
                                <div class="setup-meta-value">{_data_start} → {_data_end}</div>
                                <div class="setup-meta-note">Rozsah historických dát, na ktorých sa experiment vykonáva.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Cieľ AI</div>
                                <div class="setup-meta-value">{_goal_label}</div>
                                <div class="setup-meta-note">AI hľadá váhy portfólia v rámci zvoleného investičného profilu.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Holdout</div>
                                <div class="setup-meta-value">{FIXED_HOLDOUT_YEARS} roky</div>
                                <div class="setup-meta-note">Obdobie {_holdout_start} → {date_end} je zamknuté ako čistý nezávislý test.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Walk-forward</div>
                                <div class="setup-meta-value">{FIXED_TEST_WINDOW_MONTHS} mesiacov</div>
                                <div class="setup-meta-note">Model sa pretrénuje raz ročne a potom drží váhy počas celého testovacieho okna.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Limity</div>
                                <div class="setup-meta-value">max {int(FIXED_MAX_WEIGHT * 100)}% / aktívum</div>
                                <div class="setup-meta-note">Mesačný vklad {int(FIXED_MONTHLY_DEPOSIT)} EUR a strop koncentrácie držia simuláciu realistickú.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Jeden pokus</div>
                                <div class="setup-meta-value">1 sada nastavení</div>
                                <div class="setup-meta-note">Každý pokus znamená jednu kombináciu hyperparametrov, ktorú AI otestuje cez celé dostupné obdobie.</div>
                            </div>
                            <div class="setup-meta-card">
                                <div class="setup-meta-label">Porovnanie</div>
                                <div class="setup-meta-value">AI vs 5 referencií</div>
                                <div class="setup-meta-note">Výsledky sa porovnávajú s Equal Weight 1/N, SPY, Markowitzom, Black-Littermanom a momentum portfóliom.</div>
                            </div>
                        </div>
                        <div class="setup-section-title">Čo je v tomto profile a prečo</div>
                        <div class="setup-list-grid">
                            <ul class="setup-list">{_asset_list_left}</ul>
                            <ul class="setup-list">{_asset_list_right}</ul>
                        </div>
                    </div>
                </div>
            </details>
            """,
            unsafe_allow_html=True,
        )
        if _missing_tickers:
            st.warning(f"⚠️ Chýbajú dáta pre: `{_missing_tickers}` — budú vynechané.")

_existing_active = _list_active_scenarios(portfolio)
_existing_trials = _count_trials_in_active_scenarios(portfolio) if _existing_active else 0

# Sanity-check the holdout fit. With fixed 2013→2022 range (10y) the effective
# test range is enough for any holdout setting (0-3y) and any reasonable window.
_years_avail = (date_end - date_start).days / 365.25
_test_window_years_eff = test_window_months / 12.0
_years_needed_inside = int(holdout_years) + _test_window_years_eff
if int(holdout_years) > 0 and _years_avail < _years_needed_inside:
    st.error(
        f"❌ Rozsah {date_start} → {date_end} je krátky pre holdout="
        f"{holdout_years}r + {test_window_months} mes. okno "
        f"(potrebných min. {_years_needed_inside:.1f}r)."
    )
    st.stop()

knobs = {
    "max_weight": float(max_weight),
    "monthly_deposit_eur": float(monthly_deposit),
    "win_criterion": win_criterion,
    "date_start": str(date_start),
    "date_end": str(date_end),
    "test_window_months": int(test_window_months),
    "holdout_years": int(holdout_years),
}
# Keep holdout_years in session for cross-section UI state.
st.session_state["_current_holdout_years"] = int(holdout_years)
scenario = Scenario.from_knobs(portfolio, knobs)
trials_df = load_trials(portfolio.name, scenario.id)
# Recompute `won` against the supported comparison set (Markowitz, BL, Momentum, SPY).
# This affects display only; trials.parquet
# stores the original `won` plus `won_vs_benchmarks` / `won_vs_all` for new trials.
_baselines_for_recompute = load_baselines(portfolio.name, scenario.id)
trials_df = recompute_won_strict(trials_df, _baselines_for_recompute, win_criterion)

# ----- Slider + Start/Stop -----
st.markdown('<div id="sekcia-spustenie"></div>', unsafe_allow_html=True)
run_block = st.container(border=True)
with run_block:
    st.markdown('<div class="run-section-marker"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="run-hero">
            <div class="run-hero-icon">🧪</div>
            <div>
                <div class="run-hero-title">Spustenie testovania AI</div>
                <p class="run-hero-copy">Nastavíš počet pokusov a spustíš testovanie tvojej AI stratégie pre zvolené portfólio.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"📁 Scenár: `{scenario.id}` ({len(trials_df)} pokusov v tomto scenári)")

    # Seed-from-champion banner (if user clicked đźš€ on a champion)
    _pending_seed_hp = st.session_state.get("seed_hp")
    _pending_seed_label = st.session_state.get("seed_label")
    if _pending_seed_hp and not is_running:
        cseed_a, cseed_b = st.columns([5, 1])
        with cseed_a:
            st.markdown(
                f"<div style='background:#E3F2FD;border-left:4px solid #1976D2;"
                f"padding:10px 14px;border-radius:6px;'>"
                f"🚀 <b>Prvý pokus pôjde s hyperparametrami šampióna:</b> "
                f"<code>{_pending_seed_label}</code><br/>"
                f"<span style='font-size:0.85rem;color:#444;'>"
                f"Optuna TPE začne s týmto bodom a bude skúšať nájsť ešte lepší. "
                f"Nasledujúce pokusy = normálna TPE explorácia. "
                f"</span></div>",
                unsafe_allow_html=True,
            )
        with cseed_b:
            if st.button("✖ Zrušiť seed", use_container_width=True):
                st.session_state.pop("seed_hp", None)
                st.session_state.pop("seed_label", None)
                st.session_state.pop("seed_champion_id", None)
                st.rerun()

    # Trial count lives next to the run controls.
    if "run_n_trials_slider" not in st.session_state:
        st.session_state["run_n_trials_slider"] = int(st.session_state.get("wiz_n_trials", 25))
    _wiz_trials = int(st.session_state.get("run_n_trials_slider", st.session_state.get("wiz_n_trials", 25)))
    _wiz_trials = max(1, min(100, _wiz_trials))
    st.session_state["run_n_trials_slider"] = _wiz_trials
    is_infinite = bool(st.session_state.get("run_forever_toggle", False))
    n_trials = None if is_infinite else _wiz_trials

    top_controls_left, top_controls_right = st.columns([1, 1], vertical_alignment="top")
    with top_controls_left:
        cleanup_card = st.container(border=True)
        with cleanup_card:
            st.markdown('<div class="run-card-marker run-top-control-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="run-card-title">Vymazať staré pokusy</div>', unsafe_allow_html=True)
            if _existing_active:
                st.markdown(
                    f"<div class='run-card-copy'>Pre toto portfólio už existuje <strong>{_existing_trials}</strong> pokusov v <strong>{len(_existing_active)}</strong> aktívnych scenároch.</div>",
                    unsafe_allow_html=True,
                )
                _cleanup_status = (
                    "Pre toto portfólio práve beží hľadanie, preto teraz staré pokusy nemažeme."
                    if is_running
                    else "Ak nič nemažeš, nové pokusy sa budú prirodzene pridávať k existujúcej histórii tohto portfólia."
                )
            else:
                st.markdown(
                    "<div class='run-card-copy'>Pre toto portfólio ešte nemáš aktívne pokusy. Môžeš začať čistý experiment.</div>",
                    unsafe_allow_html=True,
                )
                _cleanup_status = "Staré pokusy tu momentálne nie sú, takže nemusíš nič čistiť pred novým spustením."
            st.markdown(f"<div class='run-card-status'>{_cleanup_status}</div>", unsafe_allow_html=True)
            if _existing_active:
                if st.button(
                    "🗑️ Vymazať staré pokusy",
                    use_container_width=True,
                    key=f"wipe_portfolio_{portfolio.name}",
                    disabled=is_running,
                ):
                    _wipe_portfolio_active_scenarios(portfolio)
                    load_trials.clear()
                    load_baselines.clear()
                    st.toast("Aktívne pokusy boli zmazané.", icon="🗑️")
                    st.rerun()
            else:
                st.button(
                    "✅ Pripravené",
                    use_container_width=True,
                    key=f"wipe_portfolio_ready_{portfolio.name}",
                    disabled=True,
                )
    with top_controls_right:
        mode_card = st.container(border=True)
        with mode_card:
            st.markdown('<div class="run-card-marker run-top-control-marker"></div>', unsafe_allow_html=True)
            st.markdown('<div class="run-card-title">Neustále trénovanie</div>', unsafe_allow_html=True)
            st.markdown(
                "<div class='run-card-copy'>Keď je zapnuté, AI bude pokračovať bez pevného limitu pokusov a slider sa uzamkne.</div>",
                unsafe_allow_html=True,
            )
            _mode_status = (
                "Počas aktívneho behu ostáva tento režim uzamknutý, aby sa nemenili podmienky experimentu."
                if is_running
                else "Zapni ho vtedy, keď chceš nechať AI hľadať nové pokusy bez vopred stanoveného limitu."
            )
            st.markdown(f"<div class='run-card-status'>{_mode_status}</div>", unsafe_allow_html=True)
            if is_running:
                st.toggle(
                    "Neustále trénovanie",
                    value=is_infinite,
                    key="run_forever_toggle_locked",
                    disabled=True,
                )
            else:
                is_infinite = st.toggle(
                    "Neustále trénovanie",
                    value=is_infinite,
                    key="run_forever_toggle",
                    help="Keď je zapnuté, AI bude pokračovať bez pevného limitu pokusov a slider sa uzamkne.",
                )
                n_trials = None if is_infinite else _wiz_trials

    trials_card = st.container(border=True)
    with trials_card:
        st.markdown('<div class="run-card-marker"></div>', unsafe_allow_html=True)
        if is_running:
            target = ctrl.get("n_trials")
            target_str = "∞ (nekonečný režim)" if target is None else f"{int(target)} pokusov"
            st.markdown('<div class="run-card-title">Počet pokusov</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="run-card-copy">Vyberte počet pokusov, ktoré sa majú vykonať.</div>',
                unsafe_allow_html=True,
            )
            st.metric("Počet pokusov v tomto behu", target_str)
            st.caption("Počas behu sa počet pokusov nemení.")
        else:
            st.markdown('<div class="run-card-title">Počet pokusov</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="run-card-copy">Vyberte pevný počet pokusov od 1 do 100. Ak je hore zapnuté neustále trénovanie, slider sa uzamkne.</div>',
                unsafe_allow_html=True,
            )
            _slider_pct = ((_wiz_trials - 1) / 99.0) * 100.0
            st.markdown(
                f"""
                <div class="run-slider-bubble-wrap">
                    <div class="run-slider-bubble" style="left:{_slider_pct:.3f}%;">{_wiz_trials}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            _n_trials_raw = st.slider(
                "Počet pokusov",
                min_value=1,
                max_value=100,
                value=_wiz_trials,
                step=1,
                key="run_n_trials_slider",
                disabled=is_infinite,
                help="Vyber pevný počet pokusov od 1 do 100. Pri neustálom trénovaní je slider uzamknutý.",
            )
            st.markdown(
                """
                <div class="run-slider-scale">
                    <span>1</span>
                    <span>10</span>
                    <span>20</span>
                    <span>30</span>
                    <span>40</span>
                    <span>50</span>
                    <span>60</span>
                    <span>70</span>
                    <span>80</span>
                    <span>90</span>
                    <span>100</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state["wiz_n_trials"] = int(_n_trials_raw)
            _wiz_trials = int(_n_trials_raw)
            n_trials = None if is_infinite else _wiz_trials

    if is_running:
        if st.button("⏸ Zastaviť", type="secondary", use_container_width=True):
            stop_auto_search(portfolio)
            st.toast("Zastavujem… loop dobehne aktuálny pokus", icon="⏸")
            time.sleep(0.5)
            st.rerun()
    else:
        _btn_label = (
            "▶ Spustiť (so seed)"
            if _pending_seed_hp
            else "▶ Spustiť pokusy"
        )
        if st.button(
            _btn_label,
            type="primary",
            use_container_width=True,
        ):
            ok, msg = start_auto_search(
                portfolio, knobs, n_trials,
                seed_hp=st.session_state.pop("seed_hp", None),
                seed_label=st.session_state.pop("seed_label", None),
            )
            st.session_state.pop("seed_champion_id", None)
            if ok:
                st.toast("Spustené!", icon="🚀")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"Spustenie zlyhalo: {msg}")

# ----- Status row -----
st.markdown('<div id="sekcia-prehlad"></div>', unsafe_allow_html=True)
results_block = st.container(border=True)
_BEST_BY_CRITERION = {
    "sharpe": ("Najlepší Sharpe", "sharpe", "{:.5f}"),
    "total_return": ("Najvyššia čistá hodnota aktív", "final_nav_eur", "{:,.0f}€"),
    "beat_benchmarks": ("Najlepší náskok", "objective", "{:+.3f}"),
}
_best_label, _best_key, _best_fmt = _BEST_BY_CRITERION.get(
    win_criterion, _BEST_BY_CRITERION["sharpe"]
)
with results_block:
    st.markdown('<div class="results-section-marker"></div>', unsafe_allow_html=True)
    st.markdown("## Prehľad výsledkov")
    st.caption(
        "Najprv vidíš stručný obraz toho, ako sa AI darí naprieč všetkými pokusmi: "
        "koľko ich už prebehlo, či AI vyhráva častejšie a či sa jej výsledky posúvajú."
    )
    status_col, count_col, best_col, won_col, trend_col = st.columns(5)
    with status_col:
        if is_running:
            target = ctrl.get("n_trials")
            completed = ctrl.get("n_completed_in_run", 0)
            target_str = "∞" if target is None else str(target)
            st.markdown(
                f"<div style='background:#E8F5E9;padding:10px;border-radius:8px;border:1px solid #A5D6A7;'>"
                f"<b>🟢 Beží</b><br/><span style='font-size:0.85rem;'>{completed} / {target_str} pokusov</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#FFF3E0;padding:10px;border-radius:8px;border:1px solid #FFCC80;'>"
                "<b>🔴 Zastavený</b></div>",
                unsafe_allow_html=True,
            )
    with count_col:
        st.metric("Pokusov celkom", f"{len(trials_df)}")
    with best_col:
        if len(trials_df) > 0 and _best_key in trials_df.columns:
            best_row = trials_df.loc[trials_df[_best_key].idxmax()]
            st.metric(_best_label, _best_fmt.format(best_row[_best_key]),
                      delta=f"#{int(best_row['trial'])}")
        else:
            st.metric(_best_label, "—")
    with won_col:
        if len(trials_df) > 0 and "won" in trials_df.columns:
            wins = int(trials_df["won"].sum())
            rate = wins / len(trials_df)
            delta_parts = [f"{rate:.0%}"]
            if "won_vs_shown" in trials_df.columns:
                wins_search = int(trials_df["won_vs_shown"].sum())
                delta_parts.append(f"search {wins_search}/{len(trials_df)}")
            delta_str = " · ".join(delta_parts)
            st.metric(
                "AI vyhrala na holdoute",
                f"{wins} / {len(trials_df)}",
                delta=delta_str,
                help="Skutočné víťazstvo: AI prekonala najlepší z 5 baseline-ov "
                     "(Equal Weight 1/N, Markowitz, Black-Litterman, Momentum, SPY) "
                     "na HOLDOUT-perióde — teda na posledných rokoch, ktoré Optuna nikdy "
                     "nevidela počas hľadania hyperparametrov. V delta vidíš počet "
                     "search-výhier (často vyšší — search výhry môžu byť overfit).",
            )
        else:
            st.metric("AI vyhrala", "—")
    with trend_col:
        # Trend: priemer posledných 10 pokusov vs predchádzajúcich 10
        if len(trials_df) >= 20 and _best_key in trials_df.columns:
            sorted_df = trials_df.sort_values("trial")
            last10 = sorted_df.tail(10)[_best_key].dropna()
            prev10 = sorted_df.iloc[-20:-10][_best_key].dropna()
            if len(last10) >= 5 and len(prev10) >= 5:
                last_mean = float(last10.mean())
                prev_mean = float(prev10.mean())
                denom = max(abs(prev_mean), 1e-6)
                pct = (last_mean - prev_mean) / denom * 100.0
                arrow = "📈" if pct > 0 else ("📉" if pct < 0 else "➡️")
                delta_str = (
                    f"Δ {last_mean - prev_mean:+,.0f}€"
                    if _best_key == "final_nav_eur"
                    else f"Î” {last_mean - prev_mean:+.3f}"
                )
                st.metric(
                    "Trend (posl. 10 vs predch. 10)",
                    f"{arrow} {pct:+.1f}%",
                    delta=delta_str,
                    help=f"Priemer kritéria '{_best_label}' za posledných 10 pokusov "
                         f"v porovnaní s predchádzajúcimi 10. Kladná hodnota = AI sa zlepšuje.",
                )
            else:
                st.metric(
                    "Trend (10/10)", "—",
                    help=f"Potrebných min. 20 pokusov ({len(trials_df)}/20). "
                         "Porovnáva priemer posledných 10 pokusov vs predchádzajúcich 10.",
                )
        else:
            st.metric(
                "Trend (10/10)", "—",
                help=f"Potrebných min. 20 pokusov ({len(trials_df)}/20). "
                     "Porovnáva priemer posledných 10 pokusov vs predchádzajúcich 10.",
            )

if len(trials_df) == 0:
    st.info(
        f"Žiadne pokusy pre **{portfolio.display_name}** zatiaľ. "
        "Klikni **▶ Spustiť pokusy**. Prvý pokus trvá ~90s, ďalšie tiež."
    )
    if is_running:
        time.sleep(8)
        st.rerun()
    st.stop()

# Prepare completed trial options once so the summary charts can jump into
# a real, fully-written trial detail below.
_all_trial_ids = (
    trials_df.sort_values("trial", ascending=False)["trial"]
    .dropna()
    .astype(int)
    .tolist()
)
trial_options = []
_incomplete_count = 0
for _tid in _all_trial_ids:
    _td = scenario.trial_dirs / f"trial_{int(_tid):04d}"
    if (_td / "returns.parquet").exists() and (_td / "walkforward.parquet").exists():
        trial_options.append(_tid)
    else:
        _incomplete_count += 1
if not trial_options:
    st.warning(
        "Žiadny pokus zatiaľ nemá kompletné výstupy. "
        "Počkaj kým prvý pokus dobehne (~15-60s)."
    )
    if is_running:
        time.sleep(8)
        st.rerun()
    st.stop()

_current_trial_choice = st.session_state.get("last_trial_choice")
if _current_trial_choice not in trial_options:
    _current_trial_choice = trial_options[0]

with results_block:
    render_learning_curve(
        trials_df=trials_df,
        best_key=_best_key,
        best_label=_best_label,
        trial_options=trial_options,
        current_trial=int(_current_trial_choice),
    )

# ----- Trial detail -----
st.markdown('<div id="sekcia-porovnanie"></div>', unsafe_allow_html=True)
compare_block = st.container(border=True)
with compare_block:
    st.markdown('<div class="compare-section-marker"></div>', unsafe_allow_html=True)
    st.markdown("## Overenie a porovnanie")
    st.caption(
        "V tejto časti si vyberieš konkrétny pokus a overíš ho cez porovnania, holdout, "
        "vývoj čistej hodnoty aktív a rizikovo-výnosové ukazovatele."
    )
    st.subheader("Vybraný pokus")
    if _incomplete_count > 0:
        st.caption(
            f"ℹ️ {_incomplete_count} pokus(ov) sa práve spracováva / nemá kompletné súbory — preskočené."
        )
# Allow chart-driven selection from the summary chart above. If a click set
# `trial_from_chart`, consume it and pre-select that trial here.
_override = st.session_state.pop("trial_from_chart", None)
if _override is not None and int(_override) in trial_options:
    _default_idx = trial_options.index(int(_override))
elif st.session_state.get("last_trial_choice") in trial_options:
    _default_idx = trial_options.index(st.session_state["last_trial_choice"])
else:
    _default_idx = 0
with compare_block:
    _sel_col, _pin_col = st.columns([4, 1])
    with _sel_col:
        chosen_trial = st.selectbox(
            "Vyber pokus",
            trial_options,
            index=_default_idx,
            format_func=lambda t: _format_trial_option(t, trials_df),
        )
        st.session_state["last_trial_choice"] = int(chosen_trial)
    with _pin_col:
        _already_pinned = is_pinned(portfolio.name, scenario.id, int(chosen_trial))
        _pin_label = "⭐ Pripnuté" if _already_pinned else "⭐ Pripnúť"
        _pin_help = (
            "Tento pokus je už uložený medzi Šampiónmi."
            if _already_pinned
            else "Uloží trial do Šampiónov pre neskoršie porovnanie alebo testovanie. "
            "keď stiahneš nové dáta a chceš znova testovať túto kombináciu hyperparametrov."
        )
        st.markdown("&nbsp;")  # vertical alignment hack
        if st.button(
            _pin_label,
            type="primary" if not _already_pinned else "secondary",
            use_container_width=True,
            help=_pin_help,
            key=f"pin_btn_{chosen_trial}",
        ):
            if _already_pinned:
                # Find this champion's id and unpin
                for ch in load_champions(portfolio.name):
                    if ch.scenario_id == scenario.id and ch.trial_id == int(chosen_trial):
                        unpin(portfolio.name, ch.id)
                        break
                st.toast("Šampión odpnutý", icon="🗑️")
            else:
                ch = pin_trial(
                    portfolio_name=portfolio.name,
                    scenario_id=scenario.id,
                    trial_id=int(chosen_trial),
                    knobs=dict(knobs),
                    label=f"{portfolio.display_name} #{chosen_trial} ({win_criterion})",
                )
                # Also archive the trial files so they survive future cleanups
                try:
                    archive_trial_files(ch)
                except Exception:
                    pass
                st.toast(f"⭐ Pripnuté ako šampión: {ch.id}", icon="⭐")
            time.sleep(0.3)
            st.rerun()

# Defensive: if duplicates slipped past load_trials dedup, take the most recent.
_trial_rows = trials_df[trials_df["trial"] == chosen_trial]
trial_row = _trial_rows.iloc[-1] if len(_trial_rows) > 0 else None
if trial_row is None:
    st.error(f"Trial #{chosen_trial} not found in trials.parquet.")
    st.stop()
trial_dir = scenario.trial_dirs / f"trial_{int(chosen_trial):04d}"
returns_path = trial_dir / "returns.parquet"

baselines = load_baselines(portfolio.name, scenario.id)
if not returns_path.exists():
    st.warning(
        f"⏳ Pokus #{chosen_trial} sa **práve spracováva** (chýba {returns_path.name}). "
        f"Počkaj ~30s a obnov stránku, alebo si vyber iný pokus zo zoznamu."
    )
    if is_running:
        time.sleep(10)
        st.rerun()
    st.stop()
if baselines is None:
    st.warning(
        "Baseline-y pre tento scenár neexistujú. Spusti aspoň 1 pokus aby sa vytvorili."
    )
    st.stop()

ai_returns = pd.read_parquet(returns_path)["returns"]
ai_returns.index.name = "date"
b_returns = baselines["returns"]
b_nav = baselines["nav"]

# Align everything on AI's index
common_idx = ai_returns.index.intersection(b_returns.index)
ai_aligned = ai_returns.reindex(common_idx)
b_returns_aligned = b_returns.reindex(common_idx)
b_nav_aligned = b_nav.reindex(common_idx)

# Compute AI clean asset value with deposits
from portopt.backtest.nav import compute_nav
from portopt.backtest.splits import monthly_rebalance_dates as _mrd

rebal = _mrd(common_idx)
nav_res = compute_nav(
    ai_aligned, rebalance_dates=rebal,
    monthly_deposit_eur=float(knobs["monthly_deposit_eur"]),
    initial_nav_eur=0.0, deposit_at_start=True,
)
ai_nav = nav_res.nav_eur

# ----- Comparison metric table -----
from portopt.evaluation.metrics import (
    alpha_beta_vs_benchmark, annualized_return, annualized_volatility,
    calmar_ratio, max_drawdown as fn_max_dd, sharpe_ratio, sortino_ratio,
)
from portopt.evaluation.metrics import total_return as fn_tr


def all_metrics(returns_series: pd.Series, nav_series: pd.Series) -> dict:
    return {
        "Čistá hodnota aktív (€)": float(nav_series.iloc[-1]),
        "Total ret": fn_tr(returns_series),
        "Ann ret": annualized_return(returns_series),
        "Ann vol": annualized_volatility(returns_series),
        "Sharpe": sharpe_ratio(returns_series),
        "Sortino": sortino_ratio(returns_series),
        "Calmar": calmar_ratio(returns_series),
        "Max DD": fn_max_dd(returns_series),
    }


rows = {"ai": all_metrics(ai_aligned, ai_nav)}
# Curated baselines only (see SHOWN_BASELINES at top of file).
for col in b_returns_aligned.columns:
    if not _is_shown_baseline(col):
        continue
    rows[col] = all_metrics(b_returns_aligned[col], b_nav_aligned[col])

metrics_df = pd.DataFrame(rows).T
metrics_df.index = [PRETTY.get(i, i) for i in metrics_df.index]

# Direction map: which way is "better"
direction = {
    "Čistá hodnota aktív (€)": 1, "Total ret": 1, "Ann ret": 1,
    "Sharpe": 1, "Sortino": 1, "Calmar": 1,
    "Ann vol": -1, "Max DD": 1,  # max_dd is negative; less-negative is better → "max" picks best
}

def style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Color top 3 rows per column using podium colors."""
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    podium_styles = [
        "background-color: #FFD54F; color: #1f2937; font-weight: 700;",
        "background-color: #CFD8DC; color: #1f2937; font-weight: 700;",
        "background-color: #D7A86E; color: #1f2937; font-weight: 700;",
    ]

    for col in df.columns:
        d = direction.get(col, 1)
        ranked = df[col].dropna().sort_values(ascending=(d == -1))
        top_idx = ranked.index[:3].tolist()
        for pos, row_name in enumerate(top_idx):
            styles.loc[row_name, col] = podium_styles[pos]

    return styles


styled_metrics = (
    metrics_df.style.format({
        "Čistá hodnota aktív (€)": "{:,.0f}",
        "Total ret": "{:.1%}", "Ann ret": "{:.1%}", "Ann vol": "{:.1%}",
        "Sharpe": "{:.5f}", "Sortino": "{:.4f}", "Calmar": "{:.4f}",
        "Max DD": "{:.1%}",
    })
    .apply(style_table, axis=None)
)

# ----- Walk-forward data (rendered later in technical details) -----
wf_path = trial_dir / "walkforward.parquet"
wf_df = pd.read_parquet(wf_path) if wf_path.exists() else None

# ----- Daily win/loss vs each shown baseline -----
# Day-by-day: AI's daily return strictly above the baseline's = "won", else "lost".
# Ties (equal return) count as losses to keep the metric conservative.
with compare_block:
    st.markdown("**📅 Denná úspešnosť AI proti baseline-om (vyhrané / prehrané dni)**")
    _dwl_cols = [c for c in b_returns_aligned.columns if _is_shown_baseline(c)]
    if _dwl_cols:
        _dwl_metric_cols = st.columns(min(len(_dwl_cols), 5))
        for _i, _b in enumerate(_dwl_cols):
            _ai_r = ai_aligned.dropna()
            _b_r = b_returns_aligned[_b].dropna()
            _common = _ai_r.index.intersection(_b_r.index)
            if len(_common) == 0:
                continue
            _diff = _ai_r.loc[_common] - _b_r.loc[_common]
            _wins = int((_diff > 0).sum())
            _losses = int((_diff <= 0).sum())
            _total = _wins + _losses
            _pct = (_wins / _total * 100.0) if _total else 0.0
            _ci_low = float("nan")
            _ci_high = float("nan")
            _p_value = float("nan")
            if _total > 0:
                _binom_res = binomtest(_wins, _total, p=0.5, alternative="two-sided")
                _ci = _binom_res.proportion_ci(confidence_level=0.95, method="wilson")
                _ci_low = float(_ci.low)
                _ci_high = float(_ci.high)
                _p_value = float(_binom_res.pvalue)
            with _dwl_metric_cols[_i % 5]:
                st.metric(
                    PRETTY.get(_b, _b),
                    f"{_wins} / {_losses}",
                    delta=f"{_pct:.1f}% vyhraných dní",
                    help=(
                        f"AI denný výnos > {PRETTY.get(_b, _b)}: {_wins}×.\n"
                        f"AI denný výnos ≤ {PRETTY.get(_b, _b)}: {_losses}×.\n"
                        f"Pomer = {_pct:.1f}% dní, v ktorých AI denne porazila tento baseline.\n"
                        f"95% Wilson CI = [{_ci_low:.1%}, {_ci_high:.1%}], p-value = {_format_p_value(_p_value)} "
                        "(binomický test proti 50 %)."
                    ),
                )
                if _total > 0:
                    st.markdown(
                        (
                            "<div style='font-size:0.78rem; color:#5d6c82; margin-top:-6px;'>"
                            f"95% CI [{_ci_low:.1%}, {_ci_high:.1%}] · p = {_format_p_value(_p_value)}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

    st.markdown("**⚖️ AI proti baseline-om: vyšší výnos a zároveň nižšie riziko**")
    if _dwl_cols and PRETTY["ai"] in metrics_df.index:
        _ai_ann_ret = float(metrics_df.loc[PRETTY["ai"], "Ann ret"])
        _ai_ann_vol = float(metrics_df.loc[PRETTY["ai"], "Ann vol"])
        _risk_wins = 0
        _risk_metric_cols = st.columns(min(len(_dwl_cols), 5))
        for _i, _b in enumerate(_dwl_cols):
            _pretty_b = PRETTY.get(_b, _b)
            if _pretty_b not in metrics_df.index:
                continue
            _b_ann_ret = float(metrics_df.loc[_pretty_b, "Ann ret"])
            _b_ann_vol = float(metrics_df.loc[_pretty_b, "Ann vol"])
            _ret_diff = _ai_ann_ret - _b_ann_ret
            _risk_diff = _b_ann_vol - _ai_ann_vol
            _beats_on_both = (_ret_diff > 0) and (_risk_diff > 0)
            if _beats_on_both:
                _risk_wins += 1
            with _risk_metric_cols[_i % 5]:
                st.metric(
                    _pretty_b,
                    "Áno" if _beats_on_both else "Nie",
                    delta=(
                        f"výnos {(_ret_diff * 100):+.1f} p. b., riziko {(_risk_diff * 100):+.1f} p. b."
                    ),
                    help=(
                        "AI vyhrá len vtedy, keď má za celé obdobie vyšší anualizovaný výnos "
                        "a zároveň nižšiu anualizovanú volatilitu než porovnávaná stratégia."
                    ),
                )
        st.caption(
            f"AI takto porazila **{_risk_wins} z {len(_dwl_cols)}** porovnávaných stratégií. "
            "Riziko je tu merané ako anualizovaná volatilita."
        )
    else:
        st.caption("Porovnanie výnosu a rizika sa zobrazí, keď budú dostupné metriky AI aj baseline-ov.")

    st.markdown("**📊 Porovnanie pokusu so všetkými baseline-mi a benchmarkmi (celé obdobie)**")
    n_trials_in_scenario = len(trials_df)
    if n_trials_in_scenario >= 3:
        st.caption(
            f"⚠️ **Tieto čísla pokrývajú celé obdobie** (search + holdout spolu). "
            f"Pre najpoctivejší out-of-sample odhad pozri sekciu **🔒 TRUE Holdout** nižšie. "
            "🥇 zlato = 1. miesto, 🥈 striebro = 2. miesto, 🥉 bronz = 3. miesto v danej kategórii."
        )
    else:
        st.caption(
            f"⚠️ Celé obdobie (search + holdout). Pre OOS odhad pozri **🔒 TRUE Holdout** sekciu nižšie. "
            "🥇🥈🥉 Farby ukazujú prvé tri miesta v každom stĺpci."
        )
    st.dataframe(
        styled_metrics,
        use_container_width=True,
        height=min(420, 38 + 35 * len(metrics_df)),
    )

# ----- TRUE Holdout (out-of-sample) section -----
from portopt.evaluation.statistics import (
    deflated_sharpe_ratio,
    diebold_mariano,
    probabilistic_sharpe_ratio,
    sharpe_bootstrap_ci,
)

# Determine holdout_start: from trial_row if present, else recompute from knobs
holdout_years_active = int(knobs.get("holdout_years", 0))
holdout_start_ts: pd.Timestamp | None = None
if "holdout_start" in trial_row.index and pd.notna(trial_row["holdout_start"]):
    try:
        holdout_start_ts = pd.Timestamp(trial_row["holdout_start"])
    except Exception:
        holdout_start_ts = None
if holdout_start_ts is None and holdout_years_active > 0:
    _end = pd.Timestamp(knobs["date_end"])
    holdout_start_ts = _end - pd.DateOffset(years=holdout_years_active) + pd.Timedelta(days=1)

# Define masks always (even if no holdout) so downstream stats section can use them
if holdout_start_ts is not None:
    search_mask = ai_aligned.index < holdout_start_ts
    holdout_mask = ai_aligned.index >= holdout_start_ts
else:
    search_mask = pd.Series(True, index=ai_aligned.index).values
    holdout_mask = pd.Series(False, index=ai_aligned.index).values

if holdout_start_ts is not None and holdout_mask.any():
    with compare_block:
        st.markdown("---")
        st.markdown(
            f"### 🔒 TRUE Holdout vyhodnotenie ({holdout_start_ts.date()} → {ai_aligned.index[-1].date()})"
        )
        st.caption(
            "Posledné **{0} roky dát** Optuna **nikdy nevidela** počas hľadania hyperparametrov. "
            "Toto je najpoctivejší out-of-sample odhad. Ak Sharpe na holdoute je výrazne nižší "
            "než na search-perióde, model bol pretrénovaný na search-dáta — "
            "**pokles nad 50 % indikuje overfit**."
            .format(holdout_years_active)
        )

    def split_metrics(ret_s: pd.Series, nav_s: pd.Series) -> dict:
        s_search = sharpe_ratio(ret_s[search_mask]) if search_mask.any() else float("nan")
        s_holdout = sharpe_ratio(ret_s[holdout_mask]) if holdout_mask.any() else float("nan")
        if pd.notna(s_search) and pd.notna(s_holdout) and abs(s_search) > 1e-6:
            gap_pct = (s_holdout - s_search) / abs(s_search) * 100.0
        else:
            gap_pct = float("nan")
        return {
            "Sharpe (holdout)": s_holdout,
            "Gap %": gap_pct,
            "Total ret (holdout)": fn_tr(ret_s[holdout_mask]) if holdout_mask.any() else float("nan"),
            "Max DD (holdout)": fn_max_dd(ret_s[holdout_mask]) if holdout_mask.any() else float("nan"),
        }

    # Keep TRUE Holdout aligned with the comparison chart and the supported baseline set.
    _holdout_compare_cols = [c for c in b_returns_aligned.columns if _is_shown_baseline(c)]
    split_rows = {"ai": split_metrics(ai_aligned, ai_nav)}
    for col in _holdout_compare_cols:
        split_rows[col] = split_metrics(b_returns_aligned[col], b_nav_aligned[col])
    split_df = pd.DataFrame(split_rows).T
    split_df.index = [PRETTY.get(i, i) for i in split_df.index]

    _WINNER_COLS = [
        "Sharpe (holdout)", "Total ret (holdout)", "Max DD (holdout)",
    ]
    _AI_ROW_NAME = PRETTY.get("ai", "ai")

    def _highlight_winner(col: pd.Series) -> list[str]:
        s = pd.to_numeric(col, errors="coerce")
        if s.dropna().empty:
            return [""] * len(col)
        best_idx = s.idxmax()  # Sharpe/Total ret: vyššie = lepšie; Max DD: záporné, max = bližšie k 0
        return [
            "background-color: #d4edda; font-weight: bold" if i == best_idx else ""
            for i in col.index
        ]

    def _highlight_ai_row(row: pd.Series) -> list[str]:
        # Jemná modrá podfarba pre celý AI riadok — komisia hneď vidí, kto je "my".
        if row.name == _AI_ROW_NAME:
            return ["background-color: #eaf1ff; font-weight: 600"] * len(row)
        return [""] * len(row)

    def _gap_color(v) -> str:
        if pd.isna(v):
            return ""
        if v >= 0:
            return "background-color: #b8e0c4; font-weight: bold"  # 🟢🟢 lepší ako search
        if v >= -15:
            return "background-color: #d4edda"   # 🟢
        if v >= -35:
            return "background-color: #fff3cd"   # 🟡
        if v >= -60:
            return "background-color: #ffe0c2"   # 🟠
        return "background-color: #e60023; color: #ffffff"  # 🔴 Pinterest red

    styled_split = (
        split_df.style
        .format({
            "Sharpe (holdout)": "{:.3f}",
            "Gap %": "{:+.1f} %",
            "Total ret (holdout)": "{:+.1%}",
            "Max DD (holdout)": "{:+.1%}",
        }, na_rep="—")
        .apply(_highlight_ai_row, axis=1)
        .apply(_highlight_winner, axis=0, subset=_WINNER_COLS)
        .map(_gap_color, subset=["Gap %"])
    )

    with compare_block:
        st.dataframe(
            styled_split,
            use_container_width=True,
            height=min(420, 38 + 35 * len(split_df)),
        )

    # Generalization gap diagnostika — rovnaké prahy ako stĺpec "Gap %"
    ai_sh_s = sharpe_ratio(ai_aligned[search_mask]) if search_mask.any() else float("nan")
    ai_sh_h = sharpe_ratio(ai_aligned[holdout_mask]) if holdout_mask.any() else float("nan")
    if pd.notna(ai_sh_s) and pd.notna(ai_sh_h) and abs(ai_sh_s) > 1e-6:
        gap_pct = (ai_sh_h - ai_sh_s) / abs(ai_sh_s) * 100.0
    else:
        gap_pct = float("nan")

    if pd.isna(gap_pct):
        gap_txt = "— search Sharpe ≈ 0, gap nedefinovaný"
    elif gap_pct >= 0:
        gap_txt = "🟢🟢 lepší na holdoute než na search — robustný model alebo náhoda"
    elif gap_pct >= -15:
        gap_txt = "🟢 minimálny prepad (do 15 %) — AI generalizuje OK"
    elif gap_pct >= -35:
        gap_txt = "🟡 mierny prepad (15–35 %) — typický gap, ešte v norme"
    elif gap_pct >= -60:
        gap_txt = (
            "🟠 stredný gap (35–60 %) — na hranici typického overfit-u; "
            "opatrná interpretácia odporúčaná "
            "(López de Prado 2018: *Advances in Financial Machine Learning*)"
        )
    else:
        gap_txt = "🔴 silný prepad (>60 %) — model takmer určite overfitnutý na search-dáta"

    # Holdout porovnanie AI vs baseline-y (Sharpe, Total ret, Max DD)
    ai_row_name = PRETTY.get("ai", "ai")
    baseline_rows = [PRETTY.get(c, c) for c in _holdout_compare_cols]
    n_baselines = len(baseline_rows)

    def _count_wins(metric: str, higher_better: bool = True) -> int:
        if ai_row_name not in split_df.index:
            return 0
        ai_v = split_df.loc[ai_row_name, metric]
        if pd.isna(ai_v):
            return 0
        wins = 0
        for b in baseline_rows:
            if b not in split_df.index:
                continue
            b_v = split_df.loc[b, metric]
            if pd.isna(b_v):
                continue
            if higher_better and ai_v > b_v:
                wins += 1
            elif (not higher_better) and ai_v < b_v:
                wins += 1
        return wins

    wins_sh = _count_wins("Sharpe (holdout)", higher_better=True)
    wins_tr = _count_wins("Total ret (holdout)", higher_better=True)
    # Max DD je záporné číslo (čím bližšie k 0, tým lepšie → AI lepšia keď AI > baseline)
    wins_dd = _count_wins("Max DD (holdout)", higher_better=True)

    with compare_block:
        gap_pct_str = "—" if pd.isna(gap_pct) else f"{gap_pct:+.1f} %"
        st.caption(
            f"**Generalization gap AI**: Sharpe search {ai_sh_s:.3f} → holdout {ai_sh_h:.3f} "
            f"({gap_pct_str}) — {gap_txt}"
        )
        st.caption(
            f"💡 **Holdout výsledok AI vs {n_baselines} baseline-y:** "
            f"Sharpe poráža **{wins_sh}/{n_baselines}**, "
            f"výnos poráža **{wins_tr}/{n_baselines}**, "
            f"lepší Max DD má voči **{wins_dd}/{n_baselines}**."
        )

@st.cache_data(ttl=600, show_spinner=False)
def _cached_stats(returns_array: tuple[float, ...], n_bootstrap: int = 800) -> dict:
    """Cache by returns tuple — small (~250-750 floats), fast hash."""
    r = pd.Series(list(returns_array))
    ci = sharpe_bootstrap_ci(r, n_bootstrap=n_bootstrap)
    return {
        "sharpe_ann": ci.sharpe_ann,
        "ci_low": ci.ci_low_ann,
        "ci_high": ci.ci_high_ann,
        "psr_0": probabilistic_sharpe_ratio(r, 0.0),
        "psr_1": probabilistic_sharpe_ratio(r, 1.0),
        "n": ci.n_observations,
    }


# Build sharpe history for DSR (all trials in this scenario, search-period sharpe)
_sharpe_history = (
    trials_df["sharpe"].dropna().astype(float).tolist()
    if "sharpe" in trials_df.columns else []
)

final_stat_trial_row = _pick_best_trial_row(trials_df, win_criterion)
final_stat_trial_id: int | None = None
final_stat_df = pd.DataFrame()
final_stat_rows: list[dict] = []
final_stat_msgs: list[str] = []
final_dm_search_msgs: list[str] = []
final_dm_holdout_msgs: list[str] = []
final_stat_note: str | None = None

if final_stat_trial_row is not None:
    try:
        final_stat_trial_id = int(final_stat_trial_row["trial"])
    except Exception:
        final_stat_trial_id = None

if final_stat_trial_id is not None:
    _final_trial_dir = scenario.trial_dirs / f"trial_{final_stat_trial_id:04d}"
    _final_returns_path = _final_trial_dir / "returns.parquet"
    if _final_returns_path.exists():
        _final_ai_returns = pd.read_parquet(_final_returns_path)["returns"]
        _final_ai_returns.index.name = "date"
        _final_common_idx = _final_ai_returns.index.intersection(b_returns.index)
        _final_ai_aligned = _final_ai_returns.reindex(_final_common_idx)
        _final_b_returns_aligned = b_returns.reindex(_final_common_idx)

        final_holdout_start_ts: pd.Timestamp | None = None
        if "holdout_start" in final_stat_trial_row.index and pd.notna(final_stat_trial_row["holdout_start"]):
            try:
                final_holdout_start_ts = pd.Timestamp(final_stat_trial_row["holdout_start"])
            except Exception:
                final_holdout_start_ts = None
        if final_holdout_start_ts is None and holdout_years_active > 0:
            _end = pd.Timestamp(knobs["date_end"])
            final_holdout_start_ts = _end - pd.DateOffset(years=holdout_years_active) + pd.Timedelta(days=1)

        if final_holdout_start_ts is not None:
            final_search_mask = _final_ai_aligned.index < final_holdout_start_ts
            final_holdout_mask = _final_ai_aligned.index >= final_holdout_start_ts
        else:
            final_search_mask = pd.Series(True, index=_final_ai_aligned.index).values
            final_holdout_mask = pd.Series(False, index=_final_ai_aligned.index).values

        _search_ret = _final_ai_aligned[final_search_mask].dropna() if hasattr(final_search_mask, "__len__") else pd.Series(dtype=float)
        _holdout_ret = _final_ai_aligned[final_holdout_mask].dropna() if (final_holdout_start_ts is not None and hasattr(final_holdout_mask, "__len__")) else pd.Series(dtype=float)

        if len(_search_ret) >= 30:
            s = _cached_stats(tuple(float(x) for x in _search_ret.values))
            dsr_s = deflated_sharpe_ratio(_search_ret, _sharpe_history) \
                if len(_sharpe_history) >= 2 else {"dsr": float("nan"), "sharpe_ref_ann": float("nan")}
            final_stat_rows.append({
                "regime": "search (Optuna videla)", "sharpe": s["sharpe_ann"],
                "CI low (95%)": s["ci_low"], "CI high (95%)": s["ci_high"],
                "PSR(>0)": s["psr_0"], "PSR(>1)": s["psr_1"],
                "DSR": dsr_s["dsr"], "n_days": s["n"],
            })

        if len(_holdout_ret) >= 30:
            s = _cached_stats(tuple(float(x) for x in _holdout_ret.values))
            dsr_h = deflated_sharpe_ratio(_holdout_ret, _sharpe_history) \
                if len(_sharpe_history) >= 2 else {"dsr": float("nan"), "sharpe_ref_ann": float("nan")}
            final_stat_rows.append({
                "regime": "holdout (TRUE OOS)", "sharpe": s["sharpe_ann"],
                "CI low (95%)": s["ci_low"], "CI high (95%)": s["ci_high"],
                "PSR(>0)": s["psr_0"], "PSR(>1)": s["psr_1"],
                "DSR": dsr_h["dsr"], "n_days": s["n"],
            })

        if final_stat_rows:
            final_stat_df = pd.DataFrame(final_stat_rows).set_index("regime")
            _s_row = final_stat_rows[0]
            _dsr_active = _s_row.get("DSR")
            if _s_row["CI low (95%)"] > 0:
                final_stat_msgs.append(
                    f"🟢 95% CI **neobsahuje 0** ({_s_row['CI low (95%)']:.2f}, {_s_row['CI high (95%)']:.2f}) — Sharpe je s 95% pravdepodobnosťou kladný."
                )
            else:
                final_stat_msgs.append(
                    f"🟡 95% CI **obsahuje 0** ({_s_row['CI low (95%)']:.2f}, {_s_row['CI high (95%)']:.2f}) — nemôžeme vylúčiť že skutočný Sharpe = 0."
                )
            if _dsr_active is not None and not pd.isna(_dsr_active):
                if _dsr_active > 0.95:
                    final_stat_msgs.append(f"🟢 DSR = {_dsr_active:.1%} — výsledok prežil korekciu na {len(_sharpe_history)} pokusov.")
                elif _dsr_active > 0.5:
                    final_stat_msgs.append(f"🟡 DSR = {_dsr_active:.1%} — slabá evidencia po korekcii na {len(_sharpe_history)} pokusov.")
                else:
                    final_stat_msgs.append(f"🔴 DSR = {_dsr_active:.1%} — výsledok po korekcii na {len(_sharpe_history)} pokusov NIE JE štatisticky významný.")

            _final_compare_cols = [c for c in _final_b_returns_aligned.columns if _is_shown_baseline(c)]
            if _final_compare_cols and len(_search_ret) >= 30:
                _b_search = _final_b_returns_aligned[_final_compare_cols].loc[final_search_mask]
                if len(_b_search) > 0:
                    for _col in _final_compare_cols:
                        _bm_s = _b_search[_col].dropna()
                        _common = _search_ret.index.intersection(_bm_s.index)
                        if len(_common) < 30:
                            continue
                        _dm = diebold_mariano(_search_ret.loc[_common], _bm_s.loc[_common], h=5)
                        _pretty = PRETTY.get(_col, _col)
                        _sig = "✅ p<0.05" if _dm["p_value"] < 0.05 else ("🟡 p<0.10" if _dm["p_value"] < 0.10 else "🔴 nevýznamné")
                        _sign = "+" if _dm["mean_diff_ann"] >= 0 else ""
                        final_dm_search_msgs.append(
                            f"**Diebold-Mariano** (finálny víťaz vs {_pretty}, search-period, h=5): "
                            f"mean diff = {_sign}{_dm['mean_diff_ann']:.2%}/rok, DM = {_dm['dm_stat']:.2f}, "
                            f"p-value = {_dm['p_value']:.3f} {_sig}"
                        )

            if _final_compare_cols and len(_holdout_ret) >= 30:
                _b_hold = _final_b_returns_aligned[_final_compare_cols].loc[final_holdout_mask]
                if len(_b_hold) > 0:
                    for _col_h in _final_compare_cols:
                        _bm_h = _b_hold[_col_h].dropna()
                        _common_h = _holdout_ret.index.intersection(_bm_h.index)
                        if len(_common_h) < 30:
                            continue
                        _dm_h = diebold_mariano(_holdout_ret.loc[_common_h], _bm_h.loc[_common_h], h=5)
                        _pretty_h = PRETTY.get(_col_h, _col_h)
                        _sig_h = "✅ p<0.05" if _dm_h["p_value"] < 0.05 else ("🟡 p<0.10" if _dm_h["p_value"] < 0.10 else "🔴 nevýznamné")
                        _sign_h = "+" if _dm_h["mean_diff_ann"] >= 0 else ""
                        final_dm_holdout_msgs.append(
                            f"**Diebold-Mariano** (finálny víťaz vs {_pretty_h}, **holdout TRUE OOS**, h=5): "
                            f"mean diff = {_sign_h}{_dm_h['mean_diff_ann']:.2%}/rok, DM = {_dm_h['dm_stat']:.2f}, "
                            f"p-value = {_dm_h['p_value']:.3f} {_sig_h}"
                        )
        else:
            final_stat_note = "Finálne overenie sa zobrazí, keď bude mať víťazný pokus dosť dát na štatistický výpočet."
    else:
        final_stat_note = "Finálne overenie sa nedá dopočítať, lebo pri víťaznom pokuse chýba súbor returns.parquet."
else:
    final_stat_note = "Finálne overenie sa zobrazí po dokončení aspoň jedného použiteľného pokusu."

# ----- Equity overlay chart -----
# Comparison chart for the supported baseline set only.
nav_card = compare_block.container()
with nav_card:
    st.markdown('<div class="nav-card-marker"></div>', unsafe_allow_html=True)
    st.markdown(f"**📈 Vývoj čistej hodnoty aktív (€) — portfólio: {portfolio.display_name}**")
_nav_chart_cols = [c for c in b_nav_aligned.columns if _is_shown_baseline(c)]
nav_full = pd.concat([ai_nav.rename("ai"), b_nav_aligned[_nav_chart_cols]], axis=1)
nav_plot = nav_full.copy()
nav_plot.index = pd.to_datetime(nav_plot.index)
nav_long = nav_plot.reset_index(names="date").melt(
    id_vars="date", var_name="strategy", value_name="nav_eur"
)
nav_long["pretty"] = nav_long["strategy"].map(PRETTY).fillna(nav_long["strategy"])

_nav_x_min = pd.Timestamp(nav_plot.index.min())
_nav_x_max = pd.Timestamp(nav_plot.index.max())
_nav_y_min_raw = float(nav_plot.min().min())
_nav_y_max_raw = float(nav_plot.max().max())
_nav_y_pad = max((_nav_y_max_raw - _nav_y_min_raw) * 0.03, max(_nav_y_max_raw, 1.0) * 0.01)
_nav_y_min = max(0.0, _nav_y_min_raw - _nav_y_pad)
_nav_y_max = _nav_y_max_raw + _nav_y_pad

_nav_phase_rows: list[dict] = []
_nav_phase_note = None
if "wf_df" in locals() and isinstance(wf_df, pd.DataFrame):
    _wf_phase = wf_df.copy()
    for _col in ["train_start", "train_end", "test_start", "test_end"]:
        if _col in _wf_phase.columns:
            _wf_phase[_col] = pd.to_datetime(_wf_phase[_col], errors="coerce")
    _wf_phase = _wf_phase.dropna(subset=[c for c in ["train_start", "test_start"] if c in _wf_phase.columns])
    if len(_wf_phase) > 0:
        _first_train_start = pd.Timestamp(_wf_phase["train_start"].min()) if "train_start" in _wf_phase.columns else None
        _first_test_start = pd.Timestamp(_wf_phase["test_start"].min()) if "test_start" in _wf_phase.columns else None

        if _first_train_start is not None and _first_test_start is not None and _first_train_start < _first_test_start:
            _nav_x_min = min(_nav_x_min, _first_train_start)
            _nav_phase_rows.append({
                "phase": "Tréning pred 1. testom",
                "date_start": _first_train_start,
                "date_end": _first_test_start,
                "y0": _nav_y_min,
                "y1": _nav_y_max,
            })

        if _first_test_start is not None:
            if holdout_start_ts is not None:
                _nav_phase_rows.append({
                    "phase": "Search test",
                    "date_start": _first_test_start,
                    "date_end": pd.Timestamp(holdout_start_ts),
                    "y0": _nav_y_min,
                    "y1": _nav_y_max,
                })
                _nav_phase_rows.append({
                    "phase": "TRUE Holdout",
                    "date_start": pd.Timestamp(holdout_start_ts),
                    "date_end": _nav_x_max + pd.Timedelta(days=1),
                    "y0": _nav_y_min,
                    "y1": _nav_y_max,
                })
                _nav_phase_note = "Pozadie: sivá = prvé tréningové obdobie, modrá = search test okná, oranžová = TRUE holdout."
            else:
                _nav_phase_rows.append({
                    "phase": "Search test",
                    "date_start": _first_test_start,
                    "date_end": _nav_x_max + pd.Timedelta(days=1),
                    "y0": _nav_y_min,
                    "y1": _nav_y_max,
                })
                _nav_phase_note = "Pozadie: sivá = prvé tréningové obdobie, modrá = testovanie v search fáze."

# User-friendly bright colors for the NAV chart.
# AI stays blue; the comparison set uses green / yellow / red tones.
_NAV_COLOR_BY_STRATEGY = {
    "ai": "#1565C0",               # modrá
    "equal_weight": "#6D4C41",     # hnedá
    "markowitz": "#2E7D32",        # zelená
    "black_litterman": "#F9A825", # žltá
    "momentum": "#E53935",         # červená
    "SPY": "#B71C1C",              # tmavšia červená
}
_FALLBACK_PALETTE = ["#1565C0", "#2E7D32", "#F9A825", "#E53935", "#B71C1C"]

_nav_color_domain = [PRETTY.get("ai", "ai")] + [PRETTY.get(c, c) for c in _nav_chart_cols]
_nav_strategy_keys = ["ai"] + list(_nav_chart_cols)
_nav_color_range = [
    _NAV_COLOR_BY_STRATEGY.get(k, _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)])
    for i, k in enumerate(_nav_strategy_keys)
]
_nav_x_scale = alt.Scale(domain=[_nav_x_min, _nav_x_max], nice=False)
_nav_y_scale = alt.Scale(domain=[_nav_y_min, _nav_y_max], nice=False, zero=False)

_nav_bg = None
_nav_phase_rules = None
if _nav_phase_rows:
    _nav_phase_df = pd.DataFrame(_nav_phase_rows)
    _nav_phase_df = _nav_phase_df[_nav_phase_df["date_end"] > _nav_phase_df["date_start"]].copy()
    if len(_nav_phase_df) > 0:
        _nav_bg = (
            alt.Chart(_nav_phase_df)
            .mark_rect(opacity=0.18)
            .encode(
                x=alt.X("date_start:T", title=None, scale=_nav_x_scale),
                x2="date_end:T",
                y=alt.Y("y0:Q", title="Čistá hodnota aktív (€)", scale=_nav_y_scale),
                y2="y1:Q",
                color=alt.Color(
                    "phase:N",
                    scale=alt.Scale(
                        domain=["Tréning pred 1. testom", "Search test", "TRUE Holdout"],
                        range=["#CFD8DC", "#BBDEFB", "#FFE0B2"],
                    ),
                    legend=None,
                ),
            )
        )
        _nav_rule_df = _nav_phase_df.iloc[1:].copy()
        if len(_nav_rule_df) > 0:
            _nav_phase_rules = (
                alt.Chart(_nav_rule_df)
                .mark_rule(color="#546E7A", strokeDash=[6, 4], strokeWidth=1.5, opacity=0.9)
                .encode(x=alt.X("date_start:T", title=None, scale=_nav_x_scale))
            )

nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)
zoom_x = alt.selection_interval(bind="scales", encodings=["x"])
selector_dates = pd.DataFrame({"date": sorted(pd.to_datetime(nav_long["date"].dropna().unique()))})

base = alt.Chart(nav_long).encode(
    x=alt.X("date:T", title=None, scale=_nav_x_scale),
    y=alt.Y("nav_eur:Q", title="Čistá hodnota aktív (€)", scale=_nav_y_scale),
    color=alt.Color(
        "pretty:N",
        scale=alt.Scale(domain=_nav_color_domain, range=_nav_color_range),
        legend=alt.Legend(
            orient="top",
            title=None,
            columns=3,
            labelFontSize=14,
            labelFontWeight=600,
            labelLimit=320,
            symbolSize=260,
            symbolStrokeWidth=3,
            symbolType="stroke",
            padding=10,
            rowPadding=8,
            columnPadding=22,
        ),
    ),
)

lines_other = (
    base.transform_filter("datum.strategy != 'ai'")
    .mark_line(strokeWidth=3.0, clip=True)
)
line_ai = (
    base.transform_filter("datum.strategy == 'ai'")
    .mark_line(strokeWidth=5.0, clip=True)
)

selectors = (
    alt.Chart(selector_dates)
    .mark_point(opacity=0)
    .encode(x=alt.X("date:T", title=None, scale=_nav_x_scale))
    .add_params(nearest)
)

hover_points = (
    base.mark_point(size=70, filled=True)
    .transform_filter(nearest)
)

hover_labels = (
    alt.Chart(nav_long)
    .mark_text(align="left", dx=7, dy=-7, fontSize=11, fontWeight="bold")
    .encode(
        x=alt.X("date:T", title=None, scale=_nav_x_scale),
        y=alt.Y("nav_eur:Q", title="Čistá hodnota aktív (€)", scale=_nav_y_scale),
        text=alt.Text("nav_eur:Q", format=",.0f"),
        color=alt.Color(
            "pretty:N",
            scale=alt.Scale(domain=_nav_color_domain, range=_nav_color_range),
            legend=None,
        ),
    )
    .transform_filter(nearest)
)

hover_rule = (
    alt.Chart(selector_dates)
    .mark_rule(color="#37474F", strokeWidth=1.4)
    .encode(x=alt.X("date:T", title=None, scale=_nav_x_scale))
    .transform_filter(nearest)
)

hover_tooltips = (
    base.mark_circle(size=220, opacity=0)
    .encode(
        tooltip=[
            alt.Tooltip("pretty:N", title="stratégia"),
            alt.Tooltip("date:T", title="dátum"),
            alt.Tooltip("nav_eur:Q", format=",.0f", title="Čistá hodnota aktív (€)"),
        ]
    )
    .transform_filter(nearest)
)

# Pridaj tenkú zvislú čiaru pri každom rebalancing-u (= start nového
# walk-forward okna). Pre 12-mesačné okná to je raz za rok.
_rebalance_marks = None
if "wf_df" in locals() and isinstance(wf_df, pd.DataFrame) and "test_start" in wf_df.columns:
    _reb_dates = (
        pd.to_datetime(wf_df["test_start"], errors="coerce")
        .dropna()
        .sort_values()
        .unique()
    )
    if len(_reb_dates) > 0:
        _reb_df = pd.DataFrame({"date": _reb_dates})
        _rebalance_marks = (
            alt.Chart(_reb_df)
            .mark_rule(color="#90A4AE", strokeDash=[2, 4], strokeWidth=1, opacity=0.7)
            .encode(x=alt.X("date:T", scale=_nav_x_scale))
        )

chart = lines_other + line_ai + selectors + hover_rule + hover_points + hover_labels + hover_tooltips
if _nav_bg is not None:
    chart = _nav_bg + chart
if _nav_phase_rules is not None:
    chart = chart + _nav_phase_rules
if _rebalance_marks is not None:
    chart = chart + _rebalance_marks
chart = chart.add_params(zoom_x).resolve_scale(color="independent").properties(height=460)
with nav_card:
    st.altair_chart(chart, use_container_width=True)
    if _nav_phase_note:
        st.caption(_nav_phase_note)
    if _rebalance_marks is not None:
        st.caption("📍 Tenké šedé prerušované čiary = momenty kedy sa AI **pretrénovala a rebalancovala** (raz ročne).")
    st.caption("🔎 Graf sa dá zoomovať po osi času: kolieskom priblížiš/oddiališ a ťahom sa posunieš do strán.")

st.markdown('<div id="sekcia-technicke"></div>', unsafe_allow_html=True)
tech_block = st.container(border=True)
with tech_block:
    st.markdown('<div class="tech-section-marker"></div>', unsafe_allow_html=True)
    st.markdown("## Technické detaily")
    render_trial_technical_sections(
        chosen_trial=int(chosen_trial),
        trial_dir=trial_dir,
        scenario=scenario,
        knobs=knobs,
        wf_df=wf_df,
    )

st.markdown('<div id="sekcia-statistika"></div>', unsafe_allow_html=True)
stats_block = st.container(border=True)
with stats_block:
    st.markdown('<div class="stats-section-marker"></div>', unsafe_allow_html=True)
    st.markdown("## Štatistická významnosť")
    st.caption(
        "Táto časť je oddelená od vybraného pokusu a počíta sa z celého experimentu. "
        "Automaticky berie finálny víťazný pokus vybraný zo všetkých dokončených pokusov v scenári."
    )
    st.caption(
        "Bodový odhad Sharpe je len odhad — pravá hodnota leží v intervale spoľahlivosti. "
        "**PSR** = pravdepodobnosť že skutočný Sharpe je > 0 (alebo > 1). "
        "**DSR** = PSR korigovaný na fakt, že sme spravili N Optuna pokusov a vybrali najlepší. "
        "**Diebold-Mariano** = test či AI má štatisticky významne vyšší výnos než porovnávaná stratégia."
    )

    st.markdown(
        """
**📐 Hypotézy testov (H₀ vs H₁)**

| Test | H₀ (nulová) | H₁ (alternatívna) | Rozhodovanie |
|---|---|---|---|
| **Bootstrap CI** | _(nie je hypotézový test, dáva 95 % interval)_ | — | Ak interval **neobsahuje 0** → môžeme zamietnuť, že Sharpe je nula |
| **PSR** (Bailey-LdP, 2012) | Skutočný Sharpe ≤ referenčný (typicky 0) | Skutočný Sharpe > referenčný | Zamietame H₀ ak **PSR > 95 %** (α = 5 % hladina významnosti); prísnejší prah **PSR > 99 %** (α = 1 %) |
| **DSR** (Bailey-LdP, 2014) | Pozorovaný max-Sharpe z N pokusov bol dosiahnutý čisto náhodou | Pozorovaný Sharpe je významne nad očakávaným max-of-N | Zamietame H₀ ak **DSR > 95 %** (α = 5 %) — model má signál nad rámec data-dredgingu |
| **Diebold-Mariano** (1995) | E[r_AI − r_baseline] = 0  (rovnaký priemerný výnos) | E[r_AI − r_baseline] ≠ 0  (rôzne priemerné výnosy) | Zamietame H₀ ak **p-value < 0.05** — rozdiel je štatisticky významný |

**Pre obhajobu:** komisia chce vidieť že **H₀ bola zamietnutá** v relevantných testoch — to znamená, že tvoja AI **NIE JE náhoda** ale štatisticky dôveryhodný výsledok.
"""
    )

if final_stat_trial_id is not None:
    with stats_block:
        st.caption(
            f"Finálny víťaz experimentu: **pokus #{final_stat_trial_id}** "
            f"podľa kritéria **{_best_label}** naprieč **{len(trials_df)} pokusmi**."
        )

if not final_stat_df.empty:
    with stats_block:
        st.dataframe(
            final_stat_df.style.format({
                "sharpe": "{:.5f}", "CI low (95%)": "{:.5f}", "CI high (95%)": "{:.5f}",
                "PSR(>0)": "{:.1%}", "PSR(>1)": "{:.1%}", "DSR": "{:.1%}", "n_days": "{:d}",
            }),
            use_container_width=True,
        )
        for p in final_stat_msgs:
            st.markdown(f"<div class='stats-emphasis'>{p}</div>", unsafe_allow_html=True)

        if final_dm_search_msgs or final_dm_holdout_msgs:
            _search_lines = "".join(
                f"<div class='stats-dm-line'>{_simple_md_to_html(msg)}</div>"
                for msg in final_dm_search_msgs
            ) or "<div class='stats-dm-line'>&nbsp;</div>"
            _holdout_lines = "".join(
                f"<div class='stats-dm-line'>{_simple_md_to_html(msg)}</div>"
                for msg in final_dm_holdout_msgs
            ) or "<div class='stats-dm-line'>&nbsp;</div>"
            _dm_grid_parts = [
                "<div class='stats-dm-grid'>",
                "<div class='stats-dm-panel'>",
                "<div class='stats-dm-heading'>Search-period</div>",
                _search_lines,
                "</div>",
                "<div class='stats-dm-panel'>",
                "<div class='stats-dm-heading'>Holdout TRUE OOS</div>",
                _holdout_lines,
                "</div>",
                "</div>",
            ]
            st.markdown("".join(_dm_grid_parts), unsafe_allow_html=True)
else:
    with stats_block:
        st.markdown(
            f"<div class='stats-emphasis'>{final_stat_note or 'Štatistická významnosť sa zobrazí, keď bude dosť dát na finálne overenie celého experimentu.'}</div>",
            unsafe_allow_html=True,
        )

# ----- Auto-refresh while running -----
if is_running:
    st.caption("⟳ Auto-refresh každých 10s")
    time.sleep(10)
    st.rerun()
