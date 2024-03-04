import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, InputExample
import faiss


cache_dir = r"C:\Users\Solmaz\.cache"
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
faiss_title_embedding = model.encode(pdf_subset.title.values.tolist())

#Train Doc2Vec model
tagged_data = [TaggedDocument(words=title.split(), tags=[str(i)]) for i, title in enumerate(pdf_subset['title'])]
doc2vec_model = Doc2Vec(vector_size=300, window=5, min_count=1, workers=4, epochs=20)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

#FAISS index
pdf_to_index = pdf_subset.set_index(["id"], drop=False)
id_index = np.array(pdf_to_index.id.values).flatten().astype("int")
content_encoded_normalized = faiss_title_embedding.copy()
faiss.normalize_L2(content_encoded_normalized)

index_content = faiss.IndexIDMap(faiss.IndexFlatIP(len(faiss_title_embedding[0])))
index_content.add_with_ids(content_encoded_normalized, id_index)



