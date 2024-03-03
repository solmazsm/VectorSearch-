;==========================================
; Title: collection.query
; Author: Solmaz Seyed Monir
; Date:   3 March 2024
;==========================================

import json

results = collection.query(query_texts=["space"], n_results=10)

print(json.dumps(results, indent=4))

collection.query(query_texts=["space"], where={"topic": "SCIENCE"}, n_results=10)
