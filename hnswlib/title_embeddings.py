;==========================================
; Title:  title_embeddings
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import pandas as pd
from sentence_transformers import SentenceTransformer
import hnswlib

cache_dir = r"C:\Users\Solmaz\.cache"
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
title_embeddings = model.encode(pdf_subset.title.values.tolist())
