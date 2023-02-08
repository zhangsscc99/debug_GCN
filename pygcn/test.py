import numpy as np
def get_adjacency_matrix(au_labels):
    # Count the occurrence of each pair of AUs in the training data set
    num_nodes = np.max(au_labels) + 1
    print(num_nodes)
    occurrence = np.zeros((num_nodes, num_nodes))
    for i in range(au_labels.shape[0]):
        for j in range(au_labels.shape[1]):
            occurrence[au_labels[i, j], au_labels[i, j]] += 1
    # Calculate the conditional probability P(Ui | Uj) for each pair of AUs
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            adjacency_matrix[i, j] = occurrence[i, j] / np.sum(occurrence[j, :])
    return adjacency_matrix

# Example usage
au_labels = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]]) # example of AUs in the training data set
#node_matrix = get_node_matrix(au_labels)
adjacency_matrix = get_adjacency_matrix(au_labels)
print(adjacency_matrix)