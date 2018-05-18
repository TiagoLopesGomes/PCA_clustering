import numpy as np
import MDAnalysis as mda
import mdtraj as md
import sys
import subprocess
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

elbow_cutoff = 0.85

class mPCA:
	"""
	This class creates an object that stores all of the quantities that need to be clustered by PCA.

	"""

	def __init__(self, PDB_list):
		"""
		Create class objects. Keep most of them empty so they can be filled as the code progresses.

		"""
		self.PDB_list = PDB_list
		self.name = []
		self.EED = [] 
		self.Rg = [] 
		self.XRg = [] 
		self.SASA = [] 
		self.Asphericity = [] 
		self.rando = []

	def plot_Kmeans_PCA(self, B, ind_list, kmeansoutput):

		plt.figure('Top 2 PCAs colored by K-Means Clustering')

		#---PCA Space
		plt.scatter(B[:,ind_list[0]], B[:,ind_list[1]], c=kmeansoutput.labels_)
		plt.xlabel('PC-1')
		plt.ylabel('PC-2')

		#---Real Space
		#plt.scatter(self.EED, self.Asphericity, c=kmeansoutput.labels_)
		#plt.xlabel('End-to-End Distance')
		#plt.ylabel('Aspherality')

		plt.title('Top 2 PCAs colored by K-Means Clustering')
		plt.savefig('PCA_kmeans.png')

	def extract_PCA(self,pca, PC_labels):
		print "PCA Variance:", pca.explained_variance_ratio_.cumsum()
		i = 1
		for pc in pca.explained_variance_ratio_.cumsum():
			if float(pc) > elbow_cutoff:
				break
			i+=1
		print "By the Elbow Rule, we will use", i, "pc's"
		nPCA = i

		comps = pca.components_[0]
		ind_list = []
		for c in range(nPCA):
			ind = np.where(comps == max(comps))[0][0]
			ind_list.append(ind)
			comps[ind] = -1

		print "Important Components:", 
		for label in range(len(ind_list)):
			print PC_labels[ind_list[label]], ",",
		print

		return ind_list

	def compute_Kmeans(self,B):
		Nc = range(1, 20)
		kmeans = [KMeans(n_clusters=i) for i in Nc]
		score = [kmeans[i].fit(B).score(B) for i in range(len(kmeans))]

		min_s = min(score)
		max_s = max(score)
		norm_score = [(s-min_s)/(max_s-min_s) for s in score]
		j = 1

		for s in norm_score:
			if s > elbow_cutoff:
				break
			j+=1

		print "By the Elbow Rule, we will use", j, "Clusters for K-Means"

		#---Plot Elbow Curve
		#plt.plot(Nc,score)
		#plt.xlabel('Number of Clusters')
		#plt.ylabel('Score')
		#plt.title('Elbow Curve')
		#plt.savefig('kmeans-elbow_curve.png')

		return j

	def norm_shift(self, vec):
		"""
		force all data to extend from -1 to 1

		"""
		vec = np.array(vec)		# [a, b]
		vec -= min(vec)			# [0, b-a]
		vec /= (max(vec) - min(vec)) 	# [0, 1]
		vec *= 2			# [0, 2]
		vec -= 1			# [-1, 1]
		return vec

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

		# to run this function, move the following lines to the "run" function
		#---My primitive implementation of PCA
		#M_PCA = self.compute_PCA(A)
		#print "mine:\n",  np.real(M_PCA).T
		#print "sklearn\n", B

		return P

	def compute_random(self):
		"""
		This is a test

		"""

		self.rando.append(np.random.uniform())
		return None
	
	def compute_Rg(self, protein):
	        """
		compute the Radius of Gyration. This is a really simple algorithm to code but I already opened MDAnalysis so 
		might as well use this.

	        """
		self.Rg.append(protein.radius_of_gyration())
		return None

	def compute_XRg(self, PDB):
		"""
		X-Ray experiments return higher values of Rg because they include some of the water in the shell. The EMBL
		Program "Crysol" computes a theoretical scattering curve for a protein and returns the Rg.

		"""

		f = PDB.split('.')[0]
		FNULL = open(os.devnull, 'w')
		subprocess.call(['crysol',f+'.pdb'], stdout=FNULL, stderr=subprocess.STDOUT)
		for line in open(f+'00.log'):
			if "Rg ( Atoms - Excluded volume + Shell ) ................. :" in line:
				self.XRg.append(float(line.split(' : ')[1]))
				os.remove(f+'00.log') ; os.remove(f+'00.alm') ; os.remove(f+'00.int')
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

	def print_results(pca):
		""" 
		this function prints tons of details 

		"""
		#---More print options
		#print "explained variance:", pca.explained_variance_
		#print "       EED              Rg              SASA              Asph"
		#print "PC-1 ", comps[0][0], comps[0][1], comps[0][2], comps[0][3]
		#print "PC-2 ", comps[1][0], comps[1][1], comps[1][2], comps[1][3]
		#print "PC-3 ", comps[2][0], comps[2][1], comps[2][2], comps[2][3]
		#print "PC-4 ", comps[3][0], comps[3][1], comps[3][2], comps[3][3]

	def run(self):
	        """
		main function within the mPCA class. Runs and handles all function calls. All data is stored in class object.

	        """
		positions = []
		#---Extract all information for each structure
		for PDB in open(PDB_list):
			PDB = PDB.split()[0]
			self.name.append(PDB)
			uni = mda.Universe(PDB)
			protein = uni.select_atoms('name CA')
			self.compute_Rg(protein)			
			self.compute_XRg(PDB)
			coors = protein.positions
			self.compute_EED(coors)
			self.compute_Asphericity(coors)
			self.compute_SASA(PDB)
			self.compute_random()

		# normalize and shift all vectors to be centered around zero
		norm_EED =self.norm_shift(self.EED)
		norm_Rg = self.norm_shift(self.Rg)
		norm_XRg = self.norm_shift(self.XRg)
		norm_SASA = self.norm_shift(self.SASA) 
		norm_Asphericity = self.norm_shift(self.Asphericity)
		norm_rando = self.norm_shift(self.rando)

		#---Prepare array containing all data
		A = np.array([norm_EED, norm_XRg, norm_SASA, norm_Asphericity]).T
		PC_labels = ['End-to-End Distance', 'X-Ray Radius of Gyration', 'SASA', 'Asphericity']

		#---Do PCA 
		pca = PCA(len(PC_labels))
		pca.fit(A)
		B = pca.transform(A)

		#---ind_list contains the important components
		ind_list = self.extract_PCA(pca,PC_labels)

		#---Do initial K-means clustering to determine number of clusters
		nK = self.compute_Kmeans(B)

		#---Use optimum number of clusters for k-means
		kmeans=KMeans(n_clusters=nK)
		kmeansoutput=kmeans.fit(np.array([B[:,ind_list[0]],B[:,ind_list[1]]]).T)

		#---Plot top 2 PCA clusters colored by kmeans
		self.plot_Kmeans_PCA(B,ind_list,kmeansoutput)

	        return None


if __name__ == "__main__":
	PDB_list = sys.argv[1]
	mPCA = mPCA(PDB_list)
	mPCA.run()
