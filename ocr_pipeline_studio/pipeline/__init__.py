"""Marks pipeline/ as a package so ``from pipeline.runner import ...`` works.

Deliberately empty of logic: the two original scripts live in this package and
importing them has side effects (PaddleOCR init, dotenv loading), so this file
must never import them at package-import time.
"""
