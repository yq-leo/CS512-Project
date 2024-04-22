import numpy as np
import torch


def get_random_anchorsets(num, c=0.5):
	"""
	Generate random anchor-sets
	:param num: number of anchor nodes
	:param c: hyperparameter for the number of anchor-sets (k=clog^2(n))
	:return: anchorsets: list of anchor-sets (idx of anchor nodes, need to re-map to the original node idx)
	"""
	m = int(np.log2(num))
	copy = int(c * m)
	anchorsets = []
	for i in range(m):
		anchor_size = int(num / np.exp2(i + 1))
		for j in range(copy):
			anchorsets.append(np.random.choice(num, size=anchor_size, replace=False))
	return anchorsets


def get_dist_max(anchorsets, dist, device):
	"""
	Get max distance & node id for each anchor-set
	:param anchorsets: list of anchor-sets
	:param dist: distance matrix (num of nodes x num of anchor nodes)
	:param device: device
	:return:
		dist_max: max distance for each anchor-set (num of nodes x num of anchor-sets)
		dist_argmax: argmax distance for each anchor-set (num of nodes x num of anchor-sets), need to re-map to the original node idx
	"""
	n, k = dist.shape[0], len(anchorsets)
	dist_max = torch.zeros((n, k)).to(device)
	dist_argmax = torch.zeros((n, k)).long().to(device)
	for i, anchorset in enumerate(anchorsets):
		anchorset = torch.as_tensor(anchorset, dtype=torch.long)
		dist_nodes_anchorset = dist[:, anchorset]
		dist_max_anchorset, dist_argmax_anchorset = torch.max(dist_nodes_anchorset, dim=-1)
		dist_max[:, i] = dist_max_anchorset
		dist_argmax[:, i] = anchorset[dist_argmax_anchorset]
	return dist_max, dist_argmax


def preselect_anchor(G1_data, G2_data, random=False, c=1, device='cpu'):
	"""
	Preselect anchor-sets
	:param G1_data: PyG Data object for graph 1
	:param G2_data: PyG Data object for graph 2
	:param random: whether to sample random anchor-sets
	:param c: hyperparameter for the number of anchor-sets (k=clog^2(n))
	:param device: device
	:return:
		dists_max: max distance for each anchor-set (num of nodes x num of anchor-sets)
		dists_argmax: argmax distance for each anchor-set (num of nodes x num of anchor-sets)
	"""
	assert G1_data.anchor_nodes.shape[0] == G2_data.anchor_nodes.shape[0], 'Number of anchor links of G1 and G2 should be the same'

	num_anchor_nodes = G1_data.anchor_nodes.shape[0]
	if random:
		anchorsets = get_random_anchorsets(num_anchor_nodes, c=c)
		G1_dists_max, G1_dists_argmax = get_dist_max(anchorsets, G1_data.dists, device)
		G2_dists_max, G2_dists_argmax = get_dist_max(anchorsets, G2_data.dists, device)
	else:
		G1_dists_max, G1_dists_argmax = G1_data.dists, torch.arange(num_anchor_nodes).repeat(G1_data.num_nodes, 1).to(device)
		G2_dists_max, G2_dists_argmax = G2_data.dists, torch.arange(num_anchor_nodes).repeat(G2_data.num_nodes, 1).to(device)

	return (G1_dists_max, G1_data.anchor_nodes[G1_dists_argmax].to(device),
			G2_dists_max, G2_data.anchor_nodes[G2_dists_argmax].to(device))

def update_anchors(out1, out2, anchor1, anchor2, detailed_loss, threshold):
	# Identify anchors with loss contributions higher than the threshold
	high_loss_indices = torch.where(detailed_loss > threshold)[0]
	
	# compute similarity based on the output
	similarity = torch.mm(out1, out2.t())

	# Placeholder: Implement your logic to select new anchors
	# For now, simply re-randomizing these for illustration
	new_anchor1 = anchor1.clone()
	new_anchor2 = anchor2.clone()

	for idx in high_loss_indices:
		# set the current similarity to negative inf
		similarity[anchor1[idx], anchor2[idx]] = -1e9

		# randomly assign anchor nodes
		# new_anchor1[idx] = torch.randint(0, out1.size(0), (1,))
		# new_anchor2[idx] = torch.randint(0, out2.size(0), (1,))

		# update based on similarity
		new_anchor2[idx] = similarity[anchor1[idx]].argmax().item()
		new_anchor1[idx] = similarity[:, anchor2[idx]].argmax().item()

	# return the new anchor nodes
	return new_anchor1, new_anchor2