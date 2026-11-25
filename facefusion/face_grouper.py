from typing import List

import numpy

from facefusion.types import Distance, Embedding, Face


def create_face_clusters(faces : List[Face], group_distance : Distance) -> List[List[int]]:
	face_clusters : List[List[int]] = []
	embeddings : List[Embedding] = []

	for face_index, face in enumerate(faces):
		face_clusters.append([ face_index ])
		embeddings.append(face.embedding_norm)

	if faces:
		face_embeddings = numpy.array(embeddings)
		cluster_distances = (1 - numpy.dot(face_embeddings, face_embeddings.T)) / 2
		numpy.fill_diagonal(cluster_distances, numpy.inf)

		for _ in range(len(faces) - 1):
			cluster_index, merge_index = numpy.unravel_index(numpy.argmin(cluster_distances), cluster_distances.shape)

			if cluster_distances[cluster_index, merge_index] < group_distance:
				cluster_index, merge_index = min(cluster_index, merge_index), max(cluster_index, merge_index)
				cluster_total = len(face_clusters[cluster_index])
				merge_total = len(face_clusters[merge_index])
				merge_distances = (cluster_distances[cluster_index] * cluster_total + cluster_distances[merge_index] * merge_total) / (cluster_total + merge_total)
				cluster_distances[cluster_index] = merge_distances
				cluster_distances[:, cluster_index] = merge_distances
				cluster_distances[cluster_index, cluster_index] = numpy.inf
				cluster_distances = numpy.delete(numpy.delete(cluster_distances, merge_index, axis = 0), merge_index, axis = 1)
				face_clusters[cluster_index].extend(face_clusters.pop(merge_index))

		for face_cluster_index, face_cluster in enumerate(face_clusters):
			face_clusters[face_cluster_index] = sort_face_cluster_by_average(face_embeddings, face_cluster)

	return sorted(face_clusters, key = len, reverse = True)


def sort_face_cluster_by_average(face_embeddings : Embedding, face_cluster : List[int]) -> List[int]:
	cluster_embeddings = face_embeddings[face_cluster]
	average_embedding = numpy.mean(cluster_embeddings, axis = 0)
	average_distances = 1 - numpy.dot(cluster_embeddings, average_embedding / numpy.linalg.norm(average_embedding))
	sort_face_cluster : List[int] = []

	for sort_index in numpy.argsort(average_distances).tolist():
		sort_face_cluster.append(face_cluster[sort_index])

	return sort_face_cluster
