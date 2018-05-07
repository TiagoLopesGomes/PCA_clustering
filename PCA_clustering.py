import numpy as np
import MDAnalysis as mda
import mdtraj as md
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class mPCA:
	"""
	This class creates an object that stores all of the quantities that need to be clustered by PCA.
	example: python PCA.py PDB_list.txt

	"""

	def __init__(self, PDB_list):
		"""
		Create class objects. Keep most of them empty so they can be filled as the code progresses.

		"""
		self.PDB_list = PDB_list
		self.name = []
		self.EED = [] 
		self.Rg = [] 
		self.SASA = [] 
		self.Asphericity = [] 
		self.reduced = []

	def run(self):
	        """
		main function within the mPCA class. Runs and handles all function calls. All data is stored in class object.

	        """
		positions = []
		for PDB in open(PDB_list):
			PDB = PDB.split()[0]
			self.name.append(PDB)
			uni = mda.Universe(PDB)
			protein = uni.select_atoms('name CA')
			self.compute_Rg(protein)			
			coors = protein.positions
			self.compute_EED(coors)
			self.compute_Asphericity(coors)
			self.compute_SASA(PDB)
		norm_EED = self.EED/np.linalg.norm(np.array(self.EED))
		norm_Rg = self.Rg/np.linalg.norm(np.array(self.Rg))
		norm_SASA = self.SASA/np.linalg.norm(np.array(self.SASA))
		norm_Asphericity = self.Asphericity/np.linalg.norm(np.array(self.Asphericity))
		A = np.array([norm_EED, norm_Rg, norm_SASA, norm_Asphericity]).T
		#M_PCA = self.compute_PCA(A)

		pca = PCA(4)
		pca.fit(A)
		B = pca.transform(A)
		#print "mine:\n",  np.real(M_PCA).T
		#print "sklearn\n", B

		print "explained variance:", pca.explained_variance_
		print pca.explained_variance_ratio_.cumsum()
		comps = pca.components_
		print "       EED              Rg              SASA              Asph"
		print "PC-1 ", comps[0][0], comps[0][1], comps[0][2], comps[0][3]
		print "PC-2 ", comps[1][0], comps[1][1], comps[1][2], comps[1][3]
		print "PC-3 ", comps[2][0], comps[2][1], comps[2][2], comps[2][3]
		print "PC-4 ", comps[3][0], comps[3][1], comps[3][2], comps[3][3]

		#print "here are the relevant columns, I guess, with the corresponding names"
		#self.reduced = np.array([self.name, B[:,0], B[:,1]]).T
		#print self.reduced

		#Nc = range(1, 20)

		#kmeans = [KMeans(n_clusters=i) for i in Nc]
		#score = [kmeans[i].fit(B).score(B) for i in range(len(kmeans))]
		#plt.plot(Nc,score)
		#plt.xlabel('Number of Clusters')
		#plt.ylabel('Score')
		#plt.title('Elbow Curve')
		#plt.savefig('test.png')

		# 8 clusters looks like it contains about 80%
		kmeans=KMeans(n_clusters=8)
		kmeansoutput=kmeans.fit(B[:,0:3])
		plt.figure('3 Cluster K-Means')

		# PCA space
		plt.scatter(B[:,0], B[:,3], c=kmeansoutput.labels_)
		plt.xlabel('PC-1')
		plt.ylabel('PC-2')

		# Real space
		#plt.scatter(self.EED, self.Asphericity, c=kmeansoutput.labels_)
		#plt.xlabel('End-to-End Distance')
		#plt.ylabel('Aspherality')
		plt.title('8 Cluster K-Means')
		#plt.show()
		plt.savefig('test.png')

	        return None

	def compute_PCA(self, A):
	        """
		perform Principle Component Analysis
		Borrowed from: https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/

	        """
		M = np.mean(A.T, axis=1)
		C = A - M
		V = np.cov(C.T)
		vectors = np.linalg.eig(V)[1]
		P = vectors.T.dot(np.transpose(C))
		return P
	
	def compute_Rg(self, protein):
	        """
		compute the Radius of Gyration. This is a really simple algorithm to code but I already opened MDAnalysis so 
		might as well use this.

	        """
		self.Rg.append(protein.radius_of_gyration())
		return None
	
	def compute_SASA(self, PDB):
	        """
		compute the Solvent Accessible Surface Area with MDTraj. The Shrake Rupley algorithm is relatively expensive
		and difficult to code, so I borrowed from MDTraj.

	        """
		struc = md.load(PDB)
		self.SASA.append(md.shrake_rupley(struc).sum(axis=1)[0])
		return None
	
	def compute_EED(self, coors):
	        """
		compute the N-terminal to C-terminal distance

	        """
		self.EED.append(np.linalg.norm(coors[0]-coors[-1]))
		return None
	
	def compute_Asphericity(self, coors):
	        """
		compute the Asphericitiy

	        """
		n = len(coors)
		COM = [sum(coors[0])/n, sum(coors[1])/n, sum(coors[2])/n]
		S = np.zeros((3,3)) # From: Gyration tensor based analysis of the shapes of polymer chains in an attractive spherical cage

		for c in coors:
			for i in range(3):
				for j in range(3):
					S[i][j] += (c[i] - COM[i]) * (c[j] - COM[j])
		S/=n
		[L1, L2, L3] = np.linalg.eig(S)[0]

		# From: Simulation Analysis of the Temperature Dependence of Lignin Structure and Dynamics
		delta = ((L1-L2)**2+(L2-L3)**2+(L1-L3)**2)/(2*(L1+L2+L3)**2)
		self.Asphericity.append(delta)
		return None

	def show(self):
		"""
		test function

		"""
		print "EED  ", len(self.EED)
		print "Rg   ", len(self.Rg)
		print "SASA ", len(self.SASA)
		print "Asph ", self.Asphericity

if __name__ == "__main__":
	PDB_list = sys.argv[1]
	mPCA = mPCA(PDB_list)
	mPCA.run()
	#mPCA.show()
