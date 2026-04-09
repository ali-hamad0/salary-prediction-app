# src/salary_predictor/db/supabase_client.py
#
# Handles the connection to Supabase.
# We create one client and reuse it everywhere —
# no need to reconnect on every function call.

import logging
import os

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger(__name__)

# Read credentials from .env — never hardcode these
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def get_client() -> Client:
    """
    Create and return a Supabase client.

    Raises:
        ValueError: if the URL or key is missing from .env
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )

    return create_client(SUPABASE_URL, SUPABASE_KEY)