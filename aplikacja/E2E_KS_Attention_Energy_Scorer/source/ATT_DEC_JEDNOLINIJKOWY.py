def atttt():

    # hidden = None ; d = dict() ; from torch import cat, bmm, exp, sum ; import torch.nn.functional as F ; enc_out = None

    # hidden: (1, N, h)
    # enc_out: (L, N, h)
    enc_out = enc_out.transpose(0, 1)
    # (N, L, h)

    scores = bmm(hidden.transpose(0, 1),enc_out.transpose(1, 2)).squeeze()
    # scores: (N,1,h)*(N,h,L) = (N,L)

    score_sum = sum(exp(bmm(enc_out, hidden.transpose(0, 1).transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2)
    # bmm: (N,L,h)*(N,h,L) = (N,L,L)
    # score_sum: sum((N,L,L), dim=2) = (N,L)

    att = exp(scores) / score_sum
    # (N, L)

    att = F.softmax(att, dim=-1).unsqueeze(1)
    # (N,1,L)

    context = bmm(att,enc_out).transpose(0, 1)
    # (N,1,L)*(N,L,h)=(N,1,h)
    # .transpose() -> (1,N,h)

    dec_in = cat((hidden, context), dim=-1)
    # (1,N,h) cat (1,N,h) = (1,N,2h)

    dec_in = d["fc_cat_to_h"](dec_in)
    # (1,N,h)

    decoder_output, hidden = d["gru_decoder"](dec_in, hidden)
    # (1,N,h)


    # ##########################################################################################################################################################################################################################################################################################
    # # wersja jednolinijkowa
    # emb, h = enc_out, hidden
    # # (L, N, h), (1, N, h)

    # emb = emb.transpose(0, 1)
    decoder_output, h = d["gru_decoder"](d["fc_cat_to_h"](cat((h, bmm(F.softmax(exp(bmm(h.transpose(0, 1),emb.transpose(1, 2)).squeeze()) / sum(exp(bmm(emb, h.transpose(0, 1).transpose(1, 2).repeat((1, 1, emb.size(1))))), dim=2), dim=-1).unsqueeze(1),emb).transpose(0, 1)), dim=-1)), h)
# # (1,N,h)
# ##########################################################################################################################################################################################################################################################################################
# # 
# # TO JEDNAK JEST TO SAMO... (bmm((N,1,L),(N,L,h)), ((N,L,1)*(N,L,h)).sum(dim=1))
# # czyli potwierdzone ze bmm oblicza elegancko sume wazona
# # VERSION 2: DIFFERENT WEIGHTED SUM WHEN CALCULATING THE CONTEXT VECTOR
# # 
# ##########################################################################################################################################################################################################################################################################################

# LOCAL ATTENTION

# hidden: (1, N, h)
# enc_out: (L, N, h)

# (1 2 3 4 5) 6 7 8

def attt():
    def get_idxs(seq_iter, len_of_seq):
        from_idx, to_idx = 0, 0
        size_of_window = len_of_seq // 5
        if seq_iter <= size_of_window // 2:
            from_idx = 0
        else:
            from_idx = seq_iter - size_of_window // 2
        if len_of_seq - seq_iter <= size_of_window // 2:
            to_idx = len_of_seq - 1
            from_idx = len_of_seq - size_of_window - 1
        else:
            to_idx = from_idx + size_of_window
        return from_idx, to_idx

    seq_iter = 0
    from_idx, to_idx = get_idxs(seq_iter, enc_out.size(0))
    local_enc_out = enc_out[from_idx:to_idx, :]

    enc_out = enc_out.transpose(0, 1)
    # (N, L, h)

    scores = bmm(hidden.transpose(0, 1),enc_out.transpose(1, 2)).squeeze()
    # scores: (N,1,h)*(N,h,L) = (N,L)

    score_sum = sum(exp(bmm(enc_out, hidden.transpose(0, 1).transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2)
    # bmm: (N,L,h)*(N,h,L) = (N,L,L)
    # score_sum: sum((N,L,L), dim=2) = (N,L)

    att = exp(scores) / score_sum
    # (N, L)

    att = F.softmax(att, dim=-1).unsqueeze(1)
    # (N,1,L)

    context = bmm(att,enc_out).transpose(0, 1)
    # (N,1,L)*(N,L,h)=(N,1,h)
    # .transpose() -> (1,N,h)

    dec_in = cat((hidden, context), dim=-1)
    # (1,N,h) cat (1,N,h) = (1,N,2h)

    dec_in = d["fc_cat_to_h"](dec_in)
    # (1,N,h)

    decoder_output, hidden = d["gru_decoder"](dec_in, hidden)
    # (1,N,h)








import torch
import torch.nn.functional as F
def att(enc_out, dec_h):
        dec_h = torch.tensor([[[1,2,3,4,5]],[[6,7,8,9,10]]], dtype=torch.float) # (2, 1, 5)
        enc_out = torch.tensor([[[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]],[[8,9,10,11,12], [9,10,11,12,13], [10,11,12,13,14]]], dtype=torch.float) # (2, 3, 5)
        dec_h /= 10
        enc_out /= 10
        
        """
        enc_out:  (N, L, h)
        dec_h:    (N, 1, h)
        returns attention weights: (N, L)
        czy softmax na koncu?
        """
        # enc_out = torch.nn.GELU()(enc_out)
        # bmm: (N,1,h)*(N,h,L) -> (N, 1, L) -> (N,L)
        # bmm: (N,L,h)*(N,h,L) -> (N, L, L) -> sum -> (N,L)
        # (N,L)/(N,L)
        print(f"\nbefore bmm:\n\tdec_h: {str(dec_h)}\n\tenc_out: {str(enc_out)}")
        score = torch.bmm(dec_h, enc_out.transpose(1, 2)).squeeze()
        scores = torch.bmm(enc_out, dec_h.transpose(
            1, 2).repeat((1, 1, enc_out.size(1))))
        print(f"\nbefore exp:\n\tscore: {str(score)}\n\tscores: {str(scores)}")
        # some max values are huge fsr
        # there are even infs...
        score = torch.exp(score)
        scores = torch.exp(scores)
        print("scores after exp: " + str(scores))
        if score.isinf().any() or scores.isinf().any():
            print("attention please: exp -> inf values >_<")
        # print(f"\after exp:\n\tscore({score.min()},{score.max()})\n\tscores({scores.min()},{scores.max()})")
        ret = score / torch.sum(scores, dim=2)

        print(torch.sum(scores, dim=2))
        print(torch.sum(scores, dim=1))

        print(ret)
        ret = F.softmax(ret, dim=-1)
        # ret = exp(bmm(dec_h, enc_out.transpose(1, 2)).squeeze(
        # )) / sum(exp(bmm(enc_out, dec_h.transpose(1, 2).repeat((1, 1, enc_out.size(1))))), dim=2)
        # print(ret.size()) # (N,L) confirmed
        return ret
        # (N, L)

def att2(enc_out, dec_h):
    # (N, L, h), (N, 1, h)
    scores = torch.sum(dec_h * enc_out, dim=-1) # (N, L)
    scores = torch.exp(scores)
    sums = torch.sum(scores, dim=-1) # (N)
    scores /= sums.unsqueeze(1).repeat(1, enc_out.size(1))
    return scores

# att(None, None)

dec_h = torch.tensor([[[1,2,3,4,5]],[[6,7,8,9,10]]], dtype=torch.float) # (2, 1, 5)
enc_out = torch.tensor([[[1,2,3,4,5], [2,3,4,5,6], [3,4,5,6,7]],[[8,9,10,11,12], [9,10,11,12,13], [10,11,12,13,14]]], dtype=torch.float) # (2, 3, 5)
enc_out = torch.tensor([[[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3]],[[8,9,10,11,12], [9,10,11,12,13], [10,11,12,13,14]]], dtype=torch.float) # (2, 3, 5)

att2(enc_out, dec_h)


# print(dec_h[0] * enc_out[0])
# print(torch.sum(dec_h * enc_out, dim=-1))
# print(torch.sum(torch.sum(dec_h * enc_out, dim=-1), dim=-1))
# print(torch.bmm(dec_h, enc_out.transpose(1,2)))

