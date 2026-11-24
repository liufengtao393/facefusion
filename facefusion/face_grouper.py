from typing import List

import numpy

from facefusion.types import Face


def create_face_groups(faces : List[Face], face_distance : float) -> List[List[int]]:
	face_groups = [ [ face_index ] for face_index in range(len(faces)) ]

	if faces:
		face_embeddings = numpy.array([ face.embedding_norm for face in faces ])
		group_distances = (1 - numpy.dot(face_embeddings, face_embeddings.T)) / 2
		numpy.fill_diagonal(group_distances, numpy.inf)

		for _ in range(len(faces) - 1):
			group_index, merge_index = numpy.unravel_index(numpy.argmin(group_distances), group_distances.shape)

			if group_distances[group_index, merge_index] < face_distance:
				group_index, merge_index = min(group_index, merge_index), max(group_index, merge_index)
				group_total = len(face_groups[group_index])
				merge_total = len(face_groups[merge_index])
				merge_distances = (group_distances[group_index] * group_total + group_distances[merge_index] * merge_total) / (group_total + merge_total)
				group_distances[group_index] = merge_distances
				group_distances[:, group_index] = merge_distances
				group_distances[group_index, group_index] = numpy.inf
				group_distances = numpy.delete(numpy.delete(group_distances, merge_index, axis = 0), merge_index, axis = 1)
				face_groups[group_index].extend(face_groups.pop(merge_index))

	return sorted(face_groups, key = len, reverse = True)


def sort_face_group_by_average(faces : List[Face]) -> List[int]:
	face_embeddings = numpy.array([ face.embedding_norm for face in faces ])
	average_embedding = numpy.mean(face_embeddings, axis = 0)
	average_distances = 1 - numpy.dot(face_embeddings, average_embedding / numpy.linalg.norm(average_embedding))
	return numpy.argsort(average_distances).tolist()
