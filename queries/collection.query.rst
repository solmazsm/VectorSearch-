.. code:: ipython3

    import json
    
    results = collection.query(query_texts=["space"], n_results=10)
    
    print(json.dumps(results, indent=4))
