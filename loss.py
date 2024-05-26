import torch


class SenseContrastLoss(torch.nn.Module):
    def __init__(self, temperature=1, sim_score='cosine'):
        super(SenseContrastLoss, self).__init__()
        self.temperature = temperature
        if sim_score == 'cosine':
            self.score = self._cos_sim
        elif sim_score == 'euclidean':
            self.score = self._euc_sim
        elif sim_score == 'dot':
            self.score = self._dot_score
        else:
            raise ValueError(f'Incorrect score name {sim_score}')

    def forward(self, vectors, labels):
        sim_matrix = torch.div(self.score(vectors, vectors), self.temperature)  # .fill_diagonal_(0)
        # remove diagonal because of exp(0) in logsumexp
        new_sim_matrix = self._remove_diagonal(sim_matrix)
        denominator_all_j = torch.logsumexp(new_sim_matrix, dim=1, keepdim=False)
        loss = 0
        loss_change = False
        for j in range(len(vectors)):
            sense_j = labels[j]
            samesense_mask = labels == sense_j
            # remove the example j itself
            samesense_mask[j] = False

            samesense_vectors = vectors[samesense_mask]
            samesense_size = samesense_vectors.size(0)
            if samesense_size == 0:
                continue
            # denominator = torch.div(denominator_all_j[j], samesense_size)
            denominator = denominator_all_j[j]

            sim_matrix_num = torch.div(self.score(vectors[j], samesense_vectors), self.temperature)
            numerator = torch.div(-sim_matrix_num.sum(), samesense_size)
            loss += torch.add(denominator, numerator)
            loss_change = True
        if not loss_change:
            print('no change')
            loss += torch.tensor(0, dtype=torch.float64, requires_grad=True)
        return loss

    @staticmethod
    def _cos_sim(a, b):
        """
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py

        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @staticmethod
    def _euc_sim(a, b):
        """
        Computes the euclidean similarity euc_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = euc_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.div(1, torch.cdist(a, b, p=2) + 1)

    @staticmethod
    def _dot_score(a, b):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))

    @staticmethod
    def _remove_diagonal(similarity_matrix):
        """
        Removes diagonal from the matrix
        :return: matrix without one element in the second dimension
        """
        mask = torch.zeros_like(similarity_matrix, dtype=bool).fill_(True).fill_diagonal_(False)
        row_size = similarity_matrix.size(0)
        new_similarity_matrix = torch.zeros((row_size, row_size - 1))
        for num_row, row in enumerate(similarity_matrix):
            new_similarity_matrix[num_row] = row[mask[num_row]]
        return new_similarity_matrix
