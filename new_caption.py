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
    beam_seqs : array containing beam sequences staggered across time
    logprobs            : log probabilities for the beam sequences
    time                : Current time unit (not adjusted for the current group
    group_number        : the current group number
    penalty_lambda      : diversity penalty
    group_size          : num_beams/num_groups
    """
    local_time = time - group_number # current time for the group
    aug_logprobs = logprobs.clone()
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
    new_state = [x.clone() for x in state]

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
        state_size,
        rnn_size,
        seq_length,
        end_token,
        gen_logprobs):
    """ Performs Diverse Beam Search
    seq_length: maximum length of sequence
    rnn_state: states of the RNNs
    logprobs: log-probabilities of the beams
    num_beams: number of beams
    num_groups: number of groups
    penalty_lambda: value of diversity penalty
    state_size: size of the RNN state
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
    for sub_beam_ix in range(beam_ratio):

    # Initialization

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
            logprobs_list[group_num][sub_beam_ix] = init_logprobs.clone()
            state[sub_beam_ix] = init_state.clone().to(device)

        states[group_num] = [st.clone().to(device) for st in state]
    # End initialization

    for time in range(seq_length + num_groups):
        for group_ix in range(num_groups):
            if time >=group_ix and time < seq_length + group_ix:
                logprobss = logprobs_list[group_ix]

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


                inpt = beam_seqs[group_ix][time - group_ix]
                output = gen_logprobs(inpt, states[group_ix])
                logprobs[group_ix] = out[-1].clone()
                temp_state = [out[i] for i in range(state_size)]
    for i in range(num_groups):
        done_beams_table[i].sort(key=lambda x:x['aug_logp'])
        done_beams_table[i] = done_beams_table[i][0:beam_ratio]
    return done_beams_table


def caption_image_beam_search(
        encoder,
        decoder,
        image_path,
        word_map,
        beam_size=3):
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
        [[word_map['<start>']]] * k
    ).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(
        k,
        1,
        enc_image_size,
        enc_image_size
    ).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    penalties = torch.zeros(seqs.size()[0], vocab_size).cuda()

    # s is a number less than or equal to k,
    #because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  #(s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)
        # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(
            decoder.f_beta(h)
        )  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(
            torch.cat([embeddings, awe], dim=1), (h, c)
        )  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        print("Scores: {0}".format(scores.size()))
        print("sequences: {0}".format(seqs.size()))


        for b1 in range(seqs.size()[0]): # Iterate over each beam:
            #B1 is the current beam we are evaluating
            for b2 in range(b1): # To compare B1 with every previous beam B2
                for token in seqs[b2]: # Check every token in b2
                    if token in seqs[b1]: # if token exists in b1 AND b2
                        # penalties[b1][:] += 1 # penalize that token
                        penalties[b1][token] += 1

        # Add diversity penalty
        # scores = scores - args._lambda * penalties
        scores = (
            top_k_scores.expand_as(scores)
            + scores
            - (args._lambda * penalties))
            / (0.7*seqs.size()[1]
        )# (s, vocab_size)

        # For the first step, all k points will have
        #the same scores (since same k previous words, h, c)

        print(scores.size())
        print(scores - args._lambda * penalties )
        if step == 1:
        top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
        # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = (
                scores.view(-1).topk(k, 0, True, True)
            )  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat(
            [seqs[prev_word_inds],
             next_word_inds.unsqueeze(1)],
            dim=1
        )  # (s, step+1)
        seqs_alpha = torch.cat(
            [seqs_alpha[prev_word_inds],
             alpha[prev_word_inds].unsqueeze(1)],
            dim=1
        )  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [
            ind
            for ind, next_word
            in enumerate(next_word_inds)
            if next_word != word_map['<end>']
        ]

        complete_inds = (
            list(
                set(range(len(next_word_inds)))
                - set(incomplete_inds)
            )
        )

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
    if step > 50:
        break
    step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visualize_att(image_path, seqs, alphas, rev_word_map, smooth=True):
                                            """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    for seq in seqs:
    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument(':img', '-i', help='path to image')
    parser.add_argument(':model', '-m', help='path to model')
    parser.add_argument(':word_map', '-wm', help='path to word map JSON')
    parser.add_argument(':beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument(':dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument(':lambda', default=0.5, type=float, dest="_lambda", help="Diversity parameter for diverse beam search")

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
    seqs, alphas = caption_image_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    for seq in seqs:
    words = [rev_word_map[ind] for ind in seq]
    print(words)

    # # Visualize caption and attention of best sequence
                                            # visualize_att(args.img, seqs, alphas, rev_word_map, args.smooth)
