import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import svd
from sklearn.manifold import TSNE

# Paths (adjust if you want)
BASE_DIR = os.path.dirname(__file__)
PH1N1_PATH = os.path.join(BASE_DIR, "pH1N1-infection.txt")
MNIST_PKL_PATH = os.path.join(BASE_DIR, "mnist-2000.pkl")

def try_read_table(path):
    """Robust read for the pH1N1 file: try common delimiters and detect gene names."""
    for sep in [',', '\t', None]:  # None -> python engine auto-detects whitespace
        try:
            df = pd.read_csv(path, sep=sep, engine='python', header=None)
            if df.shape[1] >= 11:  # expecting gene name + 10 samples
                return df
        except Exception:
            continue
    raise RuntimeError(f"Could not read {path} - please check format")

def build_X_from_ph1n1(path):
    df = try_read_table(path)
    # Robustly detect whether first column is gene names by coercing to numeric
    first_col = df.iloc[:, 0]
    coerced = pd.to_numeric(first_col, errors='coerce')
    num_numeric = coerced.notna().sum()

    if num_numeric < len(first_col) * 0.9:
        # first column is mostly non-numeric -> gene names present
        gene_names = first_col.astype(str).tolist()
        # find next 10 numeric columns after the gene name column
        numeric_cols = []
        for col in df.columns[1:]:
            colnum = pd.to_numeric(df[col], errors='coerce')
            if colnum.notna().sum() >= len(df) * 0.9:
                numeric_cols.append(col)
            if len(numeric_cols) >= 10:
                break
        if len(numeric_cols) < 10:
            raise RuntimeError("Couldn't find 10 numeric sample columns after gene name column")
        data = df[numeric_cols[:10]].astype(float).values
    else:
        # first column is numeric -> no gene name column
        gene_names = [f"gene_{i}" for i in range(df.shape[0])]
        numeric_cols = []
        for col in df.columns:
            colnum = pd.to_numeric(df[col], errors='coerce')
            if colnum.notna().sum() >= len(df) * 0.9:
                numeric_cols.append(col)
            if len(numeric_cols) >= 10:
                break
        if len(numeric_cols) < 10:
            raise RuntimeError("Couldn't find 10 numeric sample columns")
        data = df[numeric_cols[:10]].astype(float).values

    # rows = genes, cols = samples
    X = data
    return X, gene_names

def standardize_rows(X):
    means = X.mean(axis=1, keepdims=True)
    sds = X.std(axis=1, ddof=1, keepdims=True)
    sds[sds == 0] = 1.0
    return (X - means) / sds

def standardize_columns(X):
    means = X.mean(axis=0, keepdims=True)
    sds = X.std(axis=0, ddof=1, keepdims=True)
    sds[sds == 0] = 1.0
    return (X - means) / sds

def sample_space_pca(X):
    # Standardize rows (genes) so each gene has mean 0, sd 1 across samples
    Xr = standardize_rows(X)
    U, s, Vt = svd(Xr, full_matrices=False)
    m = Xr.shape[0]  # number of genes
    scale = math.sqrt(m - 1)
    # sample scores: s_k * v_k  (rows of Vt are v_k^T)
    sample_scores = (s[:, None] * Vt) / scale
    # For plotting PC1 vs PC2: rows 0 and 1 of Vt scaled by s
    pc1 = sample_scores[0, :]
    pc2 = sample_scores[1, :]
    # Variance explained
    eigenvals = (s**2) / (m - 1)
    frac1 = eigenvals[0] / eigenvals.sum()
    frac2 = eigenvals[1] / eigenvals.sum()
    return U, s, Vt, pc1, pc2, eigenvals, frac1, frac2

def gene_space_pca(X):
    # Standardize columns (samples) so each sample feature has mean 0 and sd 1 across genes
    Xc = standardize_columns(X)
    U, s, Vt = svd(Xc, full_matrices=False)
    return U, s, Vt

def plot_sample_pca(pc1, pc2, outpath):
    # first 5 infected, next 5 control
    colors = ['red'] * 5 + ['blue'] * 5
    plt.figure(figsize=(6,5))
    plt.scatter(pc1[:5], pc2[:5], c='red', label='infected', edgecolor='k')
    plt.scatter(pc1[5:], pc2[5:], c='blue', label='control', edgecolor='k')
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.title('Samples: PC1 vs PC2')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_eigensample_bar(u_col, outpath, top_n=50):
    # plot sorted U column as bar plot
    vals = u_col.copy()
    inds = np.argsort(vals)
    sorted_vals = vals[inds]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(sorted_vals)), sorted_vals)
    plt.title('First eigensample (U[:,0]) sorted values')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    # return indices for top positive/negative
    top_pos = inds[::-1][:5]
    top_neg = inds[:5]
    return top_pos, top_neg

def plot_vt_rows(Vt, outpath):
    plt.figure(figsize=(6,4))
    x = np.arange(Vt.shape[1])
    for i in range(min(3, Vt.shape[0])):
        plt.plot(x, Vt[i, :], marker='o', label=f'VT row {i+1}')
    plt.legend()
    plt.title('First three rows of V^T (eigengenes across samples)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def mnist_analysis(pkl_path, outdir):
    if not os.path.exists(pkl_path):
        print(f"MNIST pickle not found at {pkl_path}. Place mnist-2000.pkl in the folder to run MNIST parts.")
        return
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    images = np.asarray(data['images'])  # shape (2000, 784)
    labels = np.asarray(data['labels'])
    # center images (image-space PCA: observations are images, features are pixels)
    X = images - images.mean(axis=0, keepdims=True)
    # SVD with full_matrices=False
    U, s, Vt = svd(X, full_matrices=False)
    # project samples onto first two PCs
    n_images = X.shape[0]
    scale = math.sqrt(n_images - 1)
    scores = (s[:, None] * Vt) / scale  # shape (r, n_features) but we want image scores: rows of Vt? simpler use U and s
    # image scores can be computed as U * S
    img_scores = U * s  # broadcasting, shape (2000, r)
    pc1 = img_scores[:, 0]
    pc2 = img_scores[:, 1]
    # scatter colored by label
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(pc1, pc2, c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title('MNIST (2000) PCA: PC1 vs PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'mnist_pca_pc1_pc2.png'))
    plt.close()

    # Build 50-dim PCA features (images x 50)
    k = 50
    img_features_50 = img_scores[:, :k]  # (2000,50)

    # Run t-SNE for different perplexities
    for perp in [50, 25, 10, 5]:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=0, init='pca')
        Y = tsne.fit_transform(img_features_50)
        plt.figure(figsize=(7,6))
        sc = plt.scatter(Y[:,0], Y[:,1], c=labels, cmap='tab10', s=8)
        plt.colorbar(sc, ticks=range(10))
        plt.title(f't-SNE (perplexity={perp}) on first 50 PCs')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'mnist_tsne_p{perp}.png'))
        plt.close()

def main():
    # Ensure output dir
    outdir = os.path.join(BASE_DIR, 'hw4_outputs')
    os.makedirs(outdir, exist_ok=True)

    if not os.path.exists(PH1N1_PATH):
        print(f"{PH1N1_PATH} not found. Please place pH1N1-infection.txt in the directory.")
        return

    X, gene_names = build_X_from_ph1n1(PH1N1_PATH)
    print(f"Loaded X with shape {X.shape} (genes x samples) and {len(gene_names)} gene names.")

    # (b)-(d) Sample-space PCA
    U_samp, s_samp, Vt_samp, pc1, pc2, eigenvals_samp, frac1, frac2 = sample_space_pca(X)
    print(f"Sample-space PC1 variance fraction: {frac1:.4f}, PC2: {frac2:.4f}, PC1+PC2: {(frac1+frac2):.4f}")
    plot_sample_pca(pc1, pc2, os.path.join(outdir, 'samples_pc1_pc2.png'))

    # (e)-(i) Gene-space PCA
    U_gene, s_gene, Vt_gene = gene_space_pca(X)
    # (g) bar plot of first eigensample (first column of U)
    u0 = U_gene[:, 0]
    plot_eigensample_bar(u0, os.path.join(outdir, 'eigensample_u0_sorted.png'))
    # (h) find top 5 pos/neg genes
    inds_sorted = np.argsort(u0)
    top_neg_idx = inds_sorted[:5].tolist()
    top_pos_idx = inds_sorted[::-1][:5].tolist()
    print("Top 5 negative genes in PC1:", [gene_names[i] for i in top_neg_idx])
    print("Top 5 positive genes in PC1:", [gene_names[i] for i in top_pos_idx])

    # (i) plot first three rows of V^T
    plot_vt_rows(Vt_gene, os.path.join(outdir, 'vt_first3.png'))

    # MNIST parts (j)-(o)
    mnist_analysis(MNIST_PKL_PATH, outdir)

    print(f"All outputs saved in {outdir}")

if __name__ == "__main__":
    main()