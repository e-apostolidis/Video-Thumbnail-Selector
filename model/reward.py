import torch


def aesthetics_reward(aesthetic_scores, selections, num_of_picks):
    """
        Given (args):
            aesthetic_scores: a sequence of aesthetic scores [1, seq_len]
            selections: a tensor of binary values that stores data about the selected (1)
                        and non-selected frames e.g. [0, 0, 1, 0, 1]

        Compute the average aesthetic score for the collection of selected frames.

        Return:
            aes_reward: scalar that represents the aesthetics reward
    """
    aesthetic_scores = aesthetic_scores.squeeze(0)
    masked_aesthetic_scores = aesthetic_scores * selections
    total_aes_reward = torch.sum(masked_aesthetic_scores)
    aes_reward = total_aes_reward / num_of_picks

    return aes_reward
