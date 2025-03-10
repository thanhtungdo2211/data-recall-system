import numpy as np

class DiversitySampler():
    def __init__(self, K=15):
        self.K = K
            
    def k_centroid_greedy(self, dis_matrix):
        N = dis_matrix.shape[0]
        centroids = []
        c = np.random.randint(0, N, (1,))[0]
        centroids.append(c)
        i = 1
        while i < self.K:
            centroids_diss = dis_matrix[:, centroids].copy()
            centroids_diss = centroids_diss.min(axis=1)
            centroids_diss[centroids] = -1
            new_c = np.argmax(centroids_diss)
            centroids.append(new_c)
            i += 1
        return centroids

    def kmeans(self, dis_matrix, n_iter=100):
        N = dis_matrix.shape[0]
        centroids = self.k_centroid_greedy(dis_matrix)
        data_indices = np.arange(N)

        assign_dis_records = []
        for _ in range(n_iter):
            centroid_dis = dis_matrix[:, centroids]
            cluster_assign = np.argmin(centroid_dis, axis=1)
            assign_dis = centroid_dis.min(axis=1).sum()
            assign_dis_records.append(assign_dis)
            new_centroids = []
            # if len(np.unique(assign_dis)) < self.K:
            #     self.K =  len(np.unique(assign_dis))
            for i in range(self.K):
                cluster_i = data_indices[cluster_assign == i]
                assert len(cluster_i) >= 1
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                new_centroid_i = cluster_i[np.argmin(dis_mat_i.sum(axis=1))]
                new_centroids.append(new_centroid_i)
            centroids = np.array(new_centroids)
        return centroids.tolist()


    def acquisition(self, image_dis_path, uncertainty_txt):
            uncertainty_label_path = []
            with open(image_dis_path, 'rb') as frb:
                image_dis_matrix = np.load(frb)
                image_ids = np.load(frb).reshape(-1)

            with open(uncertainty_txt, 'r') as f:
                temp = f.readlines()

            if len(temp) < self.K:
                self.K = len(temp)

            centroids = self.kmeans(image_dis_matrix)
            for path in temp:
                uncertainty_label_path.append(path.strip())

            rest_image_ids = []
            for ids in image_ids:
                rest_image_ids.append(int(ids))
        
            sampled_img_ids = image_ids[centroids].tolist()
            for img_id in sampled_img_ids:
                rest_image_ids.remove(img_id)
            unsampled_img_ids = rest_image_ids
            
            sampled_path = []
            unsampled_path = []
            for sam_ids in sampled_img_ids:
                sampled_path.append(uncertainty_label_path[int(sam_ids)])
            for unsam_ids in unsampled_img_ids:
                unsampled_path.append(uncertainty_label_path[int(unsam_ids)])
            return sampled_path, unsampled_path

    def __call__(self, feature_file, uncertainty_file):
        sampled_path, unsampled_path =  self.acquisition(feature_file, uncertainty_file)
        return sampled_path, unsampled_path
