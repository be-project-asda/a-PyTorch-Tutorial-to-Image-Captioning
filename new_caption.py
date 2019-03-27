import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def __add_diversity(
        beam_seqs,
        logprobs,
        time,
        group_number,
        penalty_lambda,
        group_size):
    """ Adds a diversity penalty to a group of beams
    beam_seqs       : array containing beam sequences staggered across time
    logprobs        : log probabilities for the beam sequences
    time            : Current time unit (not adjusted for the current group
    group_number    : the current group number
    penalty_lambda  : diversity penalty
    group_size      : num_beams/num_groups
    """
    local_time = time - group_number # current time for the group
    aug_logprobs = logprobs.clone().to(device)
    for previous_choice in range(group_number):
        previous_decisions = beam_seqs[previous_choice][local_time]
        for beam in range(group_size):
            for previous_labels in range(group_size):
                aug_logprobs[beam][previous_decisions[previous_labels]] = (
                    penalty_lambda
                ) # penalize previously chosen words
    return aug_logprobs

def __beam_step(
        logprobs,
        aug_logprobs,
        beam_size,
        beam_seq,
        beam_seq_logprobs,
        beam_logprobs_sum,
        rnn_state,
        time):
    """ Runs one step of beam search
    logprobs         : log probabilities for beam_seqs
    aug_logprobs     : log probabilities after penalty
    beam_size        : Beam Size
    beam_seq         : Tensor containing the beams
    beam_seq_logprobs: log-probabilities of each beam sequence
    beam_logprobs_sum: joint log probability of each beam
    time             : time step
    """
    ys, indices = torch.sort(aug_logprobs, 2, True)
    candidates = []
    columns = min(beam_size, len(ys)[1])
    rows = beam_size
    if t == 1:
        rows = 1

    for column in range(columns):
        for row in range(rows):
            local_logprob = ys[row,column]
            candidate_logprob = beam_logprobs_sum[row] + local_logprob
            local_unaugmented_logprob = logprobs[row,indices[row][column]]
            candidates.append(
                {
                    "column":ix[row][column],
                    "row":row,
                    "prob":candidate_logprob,
                    "local_logprob":local_unaug_logprob
                }
            )
    candidates.sort(key=lambda x: x['prob'])
    new_state = [x.clone().to(device) for x in state]

    if time > 1:
        beam_seq_prev = beam_seq[0:time,:].clone()
        beam_seq_logprobs_prev = beam_seq_logprobs[0:t, :].clone()

    for v_index, candidate in enumerate(candidates):
        candidate['kept'] = True
        if time > 1:
            beam_seq[0:time, v_index] = (
                beam_seq_prev[:,candidate['row']]
            )
            beam_seq_logprobs[0:time, v_index] = (
                beam_seq_logprobs_prev[:,candidate['row']]
            )
        for state_index in range(len(new_state)):
            new_state[state_index][v_index] = (
                state[state_index][candidate['row']]
            )

        beam_seq[time,v_index] = candidate['column']
        beam_seq_logprobs[time,_v_index] = candidate['local_logprob']
        beam_logprobs_sum[v_index] = candidate['prob']
    state = new_state
    return (
        beam_seq,
        beam_seq_logprobs,
        beam_logprobs_sum,
        state,
        candidates
    )



def diverse_beam_search(
        rnn_state,
        init_logprobs,
        num_beams,
        num_groups,
        penalty_lambda,
        seq_length,
        end_token,
        gen_logprobs,
        init_token):
    """ Performs Diverse Beam Search
    seq_length: maximum length of sequence
    rnn_state: states of the RNNs
    logprobs: log-probabilities of the beams
    num_beams: number of beams
    num_groups: number of groups
    penalty_lambda: value of diversity penalty
    end_token: end-token of the vocabulary
    gen_logprobs: function that returns
    the states and logprobs from the RNN output
    """
    beam_ratio = num_beams / num_groups
    states = []
    beam_seqs = []
    beam_seq_logprobs = []
    beam_logprobs_sums = []
    done_beams = []
    logprobs_list=[]
    state=[]
    # Initialization

    state = [[] for i in range(beam_ratio)]

    for group_num in range(num_groups):
        beam_seqs[group_num] = torch.zeros(
            seq_length, beam_ratio
        ).to(device)

        beam_seq_logprobs[group_num] = torch.new_zeros(
            (seq_length, beam_ratio)
        ).to(device)

        beam_logprobs_sums[group_num] = torch.new_zeros(
            (beam_ratio)
        ).to(device)

        done_beams[group_num] = []
        logprobs_list[group_num] = torch.zeros(
            bdash, init_logprobs.size()[1]
        ).to(device)

        for sub_beam_ix in range(beam_ratio):
            logprobs_list[group_num][sub_beam_ix] = (
                init_logprobs.clone().to(device)
            )
            beam_seqs[group_num][sub_beam_ix][0] = init_token
            state[sub_beam_ix] = [st.clone().to(device) for st in init_state]

        states[group_num] = [
            [item.clone().to(device) for item in st]
            for st
            in state
        ]
    # End initialization

    for time in range(seq_length + num_groups):
        for group_ix in range(num_groups):
            if time >= group_ix and time < seq_length + group_ix:
                logprobs = logprobs_list[group_ix]

                # Suppress <UNK> tokens in the decoding
                logprobs[:,-1] -= 1000
                aug_logprobs = __add_diversity(
                    beam_seqs,
                    logprobs,
                    time,
                    group_ix,
                    penalty_lambda,
                    beam_ratio
                )

                # Runs one step of beam_search for the current group
                (
                    beam_seqs[group_ix],
                    beam_seq_logprobs[group_ix],
                    beam_logprobs_sum[group_ix],
                    states[group_ix],
                    candidates_group
                ) = __beam_step(
                        logprobs,
                        aug_logprobs,
                        beam_ratio,
                        beam_seqs[group_ix],
                        beam_seq_logprobs[group_ix],
                        beam_logprobs_sum[group_ix],
                        states[group_ix],
                        time - group_ix
                )

                for beam_ix in beam_ratio:
                    is_first_end_token = (
                        (
                            beam_seqs[group_ix][:,beam_ix][time-group_ix] == \
                         end_token
                        ) and (
                            torch.eq(
                                beam_seqs[group_ix][:,beam_ix],
                                end_token
                            ).sum() == 0
                        )
                    )

                    final_time_wo_end = (
                        (
                            time == seq_length + group_ix
                        ) and (
                            torch.eq(
                                beam_seqs[group_ix][:,beam_ix],
                                end_token
                            ).sum() == 0
                        )
                    )

                    if is_first_end_token or final_time_wo_end:
                        final_beam = {
                            "seq": beam_seqs[group_ix][:,beam_ix].clone(),
                            "logps": beam_seq_logprobs[group_ix]\
                                [:,beam_ix].clone(),
                            "logp": beam_seq_logprobs[group_ix][:,beam_ix]\
                                .sum(),
                            "aug_logp": beam_logprobs_sum[group_ix][beam_ix]
                        }
                        final_beam["candidate"] = candidates_group[beam_ix]
                        done_beams[beam_ix] = final_beam

                    if is_first_end_token:
                        beam_logprobs_sum[group_ix][beam_ix] = -1000

                inpt = beam_seqs[group_ix]
                output = gen_logprobs(inpt, states[group_ix])
                logprobs[group_ix] = out[-1].clone()
                temp_state = [
                    out[i].clone().to(device)
                    for i
                    in range(len(output)-1)
                ]
                states[group_ix] = [st.clone().to(device) for st in temp_state]

    for i in range(num_groups):
        done_beams[i].sort(key=lambda x:x['aug_logp'])
        done_beams[i] = done_beams[i][0:beam_ratio]
    return done_beams




def caption_image_beam_search(
        encoder,
        decoder,
        image_path,
        word_map,
        beam_size=8,
        group_size=8
        penalty):
    """Reads an image and captions it with beam search.
    param encoder: encoder model
    param decoder: decoder model
    param image_path: path to image
    param word_map: word map
    param beam_size: number of sequences to consider at each decode-step
    return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([normalize])
        image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)
                # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    # (1, num_pixels, encoder_dim)

    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(
        k,
        num_pixels,
        encoder_dim
    )  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step;
    #now they're just <start>
    k_prev_words = torch.LongTensor(
        [[word_map['<start>']]]
    ).to(device)  # (k, 1)






    h, c = decoder.init_hidden_state(encoder_out)

    def generate_log_probabilities(inputs, states):
        h_ = states[0]
        c_ = states[1]
        embeddings = decoder.embedding(inputs).squeeze(1)
        awe, alpha = decoder.attention(encoder_out, h_)
        gate = decoder.sigmoid(decoder.f_beta(h_))
        awe = gate * awe
        h_, c_ = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1),
            (h_, c_)
        )
        scores = F.log_softmax(decoder.fc(h_), dim=1)
        return [h_, c_, scores]

    done_beams = diverse_beam_search(
        rnn_state=[h, c],
        init_logprobs,
        num_beams=k,
        num_groups=g,
        penalty_lambda=penalty,
        seq_length=50,
        end_token=word_map['<end>'],
        gen_logprobs=generate_log_probabilities,
        init_token=word_map['<start>']
    )
    # s is a number less than or equal to k,
    #because sequences are removed from this process once they hit <end>
    seq = [beam['seq'] for beam in done_beams]
    return seq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Show, Attend, and Tell - Tutorial - Generate Caption'
    )
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument(
        '--beam_size',
        '-b',
        dest="beam_size",
        default=8,
        type=int,
        help='beam size for beam search'
    )

    parser.add_argument(
        '--group_size',
        '-b',
        dest="group_size",
        default=8,
        type=int,
        help='beam size for beam search'
    )
    parser.add_argument(
        '--dont_smooth',
        dest='smooth',
        action='store_false',
        help='do not smooth alpha overlay'
    )
    parser.add_argument(
        '--lambda',
        default=0.5,
        type=float,
        dest="_lambda",
        help="Diversity parameter for diverse beam search"
    )

    args = parser.parse_args()
    # Load model
    checkpoint = torch.load(args.model)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
    word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Encode, decode with attention and beam search
    seqs = caption_image_beam_search(
        encoder,
        decoder,
        image_path,
        word_map,
        beam_size=args.beam_size,
        group_size=args.group_size,
        penalty=args._lambda
    )
    for seq in seqs:
    words = [rev_word_map[ind] for ind in seq]
    print(words)

    # # Visualize caption and attention of best sequence
    # visualize_att(args.img, seqs, alphas, rev_word_map, args.smooth)
