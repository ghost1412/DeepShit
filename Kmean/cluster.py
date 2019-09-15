import imageio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KMeans:

    def __init__(self, max_iter=300, n_iter=10, n_clusters=3, tolerence=1e-5):
        self.n_clusters = n_clusters
        self.tolerence = tolerence
        self.max_iter = max_iter
        self.n_iter = n_iter

    def fit(self, X):

        print()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.title('Clustring')
        writer = imageio.get_writer('cluster.gif', mode='I', duration=0.5)

        min_error = 1e8

        # Iterate for convergence
        for n_iter in range(self.n_iter):


            print('Iteration:', n_iter)
            
            # Intialize  Randomly Centroids
            centroids = X[np.random.choice(X.shape[0], self.n_clusters)]

            # Iteratively Minimize Variance
            for iter in range(self.max_iter):

                # Classify based on previous centroids
                classification = np.argmin(np.linalg.norm(X[:, np.newaxis]-centroids, axis=2), axis=1)
                prev_centroids = np.copy(centroids)

                # Update Centroids by taking mean over new clusters
                for k in range(self.n_clusters):
                    mask = classification==k
                    if np.count_nonzero(mask) == 0: continue
                    centroids[k] = np.mean(X[mask], axis=0)

                # Update the steps in graph
                max_diff = np.max(np.linalg.norm(centroids-prev_centroids, axis=0))
                print('\r[', iter, '] Maximum Difference: ', round(max_diff, 4), sep='', end='   ', flush=True)

                handles = []
                # Plot Clusters
                for l in range(self.n_clusters):
                    L = X[classification==l]
                    handles.append(ax.scatter(L[:, 0], L[:, 1], L[:, 2], 
                        marker='.', c='C{}'.format(l), label='Cluster {}'.format(l)))

                # Plot Centroids
                handles.append(ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                    c='#000000', marker='D', label='Centroids'))
                
                labels = []
                for i in range(self.n_clusters):
                    labels.append('Cluster ' + str(i+1))
                labels.append('Centroids')
                plt.legend(tuple(handles), tuple(labels))
                txt = 'RUN ' + str(n_iter+1) + '\nIteration: ' + str(iter)
                text = fig.text(.5, .05, txt, ha='center')
                
                # Update GIF
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.draw()
                plt.savefig('temp'+str(iter)+'.jpg')
                image = imageio.imread('temp'+str(iter)+'.jpg')
                writer.append_data(image)
                for x in handles: x.remove()
                text.remove()
                    
                # break if max diff less than tolereance
                if max_diff < self.tolerence:
                    print(' \nCentroid shift less than tolerence so breaking:', end='')
                    break
            
            # Keeping best centroids
            distance = np.zeros(X.shape[0])
            for k in range(self.n_clusters):
                distance[classification == k] = np.linalg.norm(X[classification == k] - centroids[k], axis=1)
            error =  np.sum(np.square(distance))

            if error < min_error:
                min_error = error
                self.centroids = centroids

    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis]-self.centroids, axis=2), axis=1)

if __name__ == "__main__":
    #generate random data
    X = np.random.rand(4000,3)
    k = KMeans()
    k.fit(X)
    k.predict(X)
