from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df)
df_pca_all = pca.transform(df)
eigenvalues = pca.explained_variance_
plt.bar(np.arange(0,df.shape[1],1), eigenvalues)
plt.plot(eigenvalues, "r")
plt.plot(eigenvalues, "ro")
plt.show()
