"""
Rate limiting via slowapi (per-client-IP token bucket).
Kept in its own module so main.py stays focused on routing.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

from app import config

limiter = Limiter(key_func=get_remote_address, default_limits=[config.RATE_LIMIT])