"""Marks pipeline/ as a package.

Deliberately empty: the original scripts live here and importing them has side
effects (PaddleOCR init, dotenv loading), so this file must never pull them in
at package-import time.
"""
