# vec_store_list.py
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()  # uses OPENAI_API_KEY from env

resp = client.vector_stores.list(limit=50)
for vs in getattr(resp, "data", []):
    # name may be None if you didn’t set one
    print(f"ID={vs.id}  NAME={getattr(vs, 'name', None)}  CREATED={getattr(vs, 'created_at', None)}")
