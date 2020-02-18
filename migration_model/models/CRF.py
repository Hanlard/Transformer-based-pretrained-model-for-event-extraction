# @Author : bamtercelboo
# @Datetime : 2018/9/14 9:51
# @File : CRF.py
# @Last Modify Time : 2018/9/14 9:51
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  CRF.py
    FUNCTION : None
    REFERENCE : https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py
"""
import torch
from torch.autograd.variable import Variable
import torch.nn as nn


def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


class CRF(nn.Module):
    """
        CRF
    """
    def __init__(self, **kwargs):
        """
        kwargs:
            target_size: int, target size
            device: str, device
        """
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        device = self.device

        # init transitions
        self.START_TAG, self.STOP_TAG = -2, -1
        init_transitions = torch.zeros(self.target_size + 2, self.target_size + 2, device=device)
        init_transitions[:, self.START_TAG] = -10000.0
        init_transitions[self.STOP_TAG, :] = -10000.0
        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        """ be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1) """
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)
        """ need to consider start """
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        """ only need start from start_tag """
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        """
        add start score (from start to all tag, duplicate to batch_size)
        partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        iter over last scores
        """
        for idx, cur_values in seq_iter:
            """
            previous to_target is current from_target
            partition: previous results log(exp(from_target)), #(batch_size * from_target)
            cur_values: bat_size * from_target * to_target
            """
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            """ effective updated partition part, only keep the partition value of mask value = 1 """
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            """ let mask_idx broadcastable, to disable warning """
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            """ replace the partition where the maskvalue=1, other partition value keeps the same """
            partition.masked_scatter_(mask_idx, masked_cur_partition)
        """ 
        until the last state, add transition score for all partition (and do log_sum_exp) 
        then select the value in STOP_TAG 
        """
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.STOP_TAG]
        return final_partition.sum(), scores

    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        # print(feats.size())
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # assert(tag_size == self.tagset_size+2)
        """ calculate sentence length for each sentence """
        length_mask = torch.sum(mask.long(), dim=1).view(batch_size, 1).long()
        """ mask to (seq_len, batch_size) """
        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size
        """ be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1) """
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        """ need to consider start """
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        # record the position of best score
        back_points = list()
        partition_history = list()
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask = (1 - mask.long()).byte()
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        """ only need start from start_tag """
        partition = inivalues[:, self.START_TAG, :].clone().view(batch_size, tag_size)  # bat_size * to_target_size
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            """
            previous to_target is current from_target
            partition: previous results log(exp(from_target)), #(batch_size * from_target)
            cur_values: batch_size * from_target * to_target
            """
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            """ forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG """
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            """
            cur_bp: (batch_size, tag_size) max source score position in current tag
            set padded label as 0, which will be filtered in post processing
            """
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)
        """ add score to final STOP_TAG """
        partition_history = torch.cat(partition_history, 0).view(seq_len, batch_size, -1).transpose(1, 0).contiguous() ## (batch_size, seq_len. tag_size)
        """ get the last position for each setences, and select the last partitions using gather() """
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        """ calculate the score from last partition to end state (and then select the STOP_TAG from it) """
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size, device=self.device, requires_grad=True).long()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        """ elect end ids in STOP_TAG """
        pointer = last_bp[:, self.STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        """move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values """
        back_points.scatter_(1, last_position, insert_last)
        back_points = back_points.transpose(1,0).contiguous()
        """ decode from the end, padded position ids are 0, which will be filtered if following evaluation """
        # decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
        decode_idx = torch.empty(seq_len, batch_size, device=self.device, requires_grad=True).long()
        decode_idx[-1] = pointer.detach()
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.detach().view(batch_size)
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        """
        :param feats:
        :param mask:
        :return:
        """
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        # print(scores.size())
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)
        tags = tags.view(batch_size, seq_len)
        """ convert tag value into a new format, recorded label bigram information to index """
        # new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        new_tags = torch.empty(batch_size, seq_len, device=self.device, requires_grad=True).long()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]

        """ transition for label to STOP_TAG """
        end_transition = self.transitions[:, self.STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        """ length for batch,  last word position = length - 1 """
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        """ index the label id of last word """
        end_ids = torch.gather(tags, 1, length_mask-1)

        """ index the transition score for end_id to STOP_TAG """
        end_energy = torch.gather(end_transition, 1, end_ids)

        """ convert tag as (seq_len, batch_size, 1) """
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        """ need convert tags id to search from 400 positions of scores """
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        """
        add all score together
        gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        """
        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        return forward_score - gold_score



