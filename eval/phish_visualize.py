import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

import os

parser = argparse.ArgumentParser("phishing_visual")
parser.add_argument("--data_dir", type=str, default=None, help="the input directory of address and embedding list")
args = parser.parse_args()

phisher_account_set = set()
with open("../data/phisher_account.txt", "r") as f:
    for line in f.readlines():
        phisher_account_set.add(line.strip())

input_dir = f"../outputs/{args.data_dir}"
address_input_dir = os.path.join(input_dir, "address.npy")
embed_input_dir = os.path.join(input_dir, "embedding.npy")

address_list = np.load(address_input_dir)
X = np.load(embed_input_dir)

y = []
for addr in address_list:
    y.append(1 if addr in phisher_account_set else 0)

# visualization 준비
y = np.array(y)
p_idx = np.where(y == 1)[0]
n_idx = np.where(y == 0)[0]
X_phisher = X[p_idx]
X_normal = X[n_idx]

permutation = np.random.permutation(len(X_normal))
X_normal_sample = X_normal[permutation[:10000]]
X4tsne = np.concatenate([X_normal_sample, X_phisher], axis=0)

tsne = TSNE(n_components=2, init="random")
X_tsne = tsne.fit_transform(X4tsne)

# 첫 번째 플롯 (phisher 포함)
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(X_tsne[:10000, 0], X_tsne[:10000, 1], marker=".")
plt.scatter(X_tsne[10000:, 0], X_tsne[10000:, 1], marker=".", color="orange")
plt.savefig(f"../outputs/img/{args.data_dir}_with_phisher.png", dpi=300)
plt.close()

# 두 번째 플롯 (normal만)
plt.figure(figsize=(8, 6), dpi=80)
plt.scatter(X_tsne[:10000, 0], X_tsne[:10000, 1], marker=".")
plt.savefig(f"../outputs/img/{args.data_dir}_normal_only.png", dpi=300)
plt.close()

print("TSNE 시각화 결과 저장 완료: tsne_with_phisher.png, tsne_normal_only.png")