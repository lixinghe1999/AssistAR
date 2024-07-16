import yaml
import torch
yaml_file = 'egoobjects.yaml'
yaml_dict = yaml.load(open(yaml_file, 'r'), Loader=yaml.FullLoader)
names = yaml_dict['names'].values()

from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define a function to get the word embedding
def get_word_embedding(word):
    # Tokenize the word
    token = tokenizer.encode_plus(word, return_tensors='pt')
    
    # Get the word embedding
    with torch.no_grad():
        output = model(**token)
        word_embedding = output.last_hidden_state[0, 0, :]
    
    return word_embedding.tolist()

# Example list of object names
object_names = ['countertop', 'table', 'chair', 'sofa', 'bed', 'dresser', 'nightstand']
object_names = list(names)
# Get the word embeddings for the object names
embeddings = np.array([get_word_embedding(name) for name in object_names])


tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(embeddings)

# Create a scatter plot of the embeddings
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for i, name in enumerate(object_names):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], name, fontsize=10)

plt.title('Visualization of Object Name Clustering')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig('figs/cluster.png')