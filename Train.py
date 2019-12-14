from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Classes import *
from Proc import Proc

import os
import random
from torch import optim


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

proc = Proc(10,3)

corpus_name = "train"
corpus = os.path.join("Data", corpus_name)

datafile = os.path.join(corpus, "di_all.txt")

PAD_token = 0
SOS_token = 1
EOS_token = 2

MAX_LENGTH = 10
MIN_COUNT = 3


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, flag = True):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    if flag:
        loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, val, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    training_batches = [proc.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]
    val_batches = [proc.batch2TrainData(voc, [random.choice(val) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    print_loss_val = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        val_batch = val_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        input_variable, lengths, target_variable, mask, max_target_len = val_batch
        loss_val = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                         decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, flag=False)
        print_loss_val += loss_val

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print_loss_val_avg = print_loss_val / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Loss_val: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg, print_loss_val_avg))
            print_loss = 0
            print_loss_val = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

if __name__ == '__main__':
    save_dir = os.path.join("Data", "save")
    voc, pairs = proc.loadPrepareData(corpus, corpus_name, datafile, save_dir)
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
        print('\n')

    pairs = proc.trimRareWords(voc, pairs)

    # voc.save()
    # lines_tr = open("Data/train/di_train_95.txt", encoding='utf-8').read().strip().split('\n')
    # lines = open("Data/train/di_val_5.txt", encoding='utf-8').read().strip().split('\n')
    # tr_pairs = [[proc.normalizeString(s) for s in l.split('\t')] for l in lines_tr]
    # val_pairs = [[proc.normalizeString(s) for s in l.split('\t')] for l in lines]
    # val = proc.filterPairs(val_pairs)
    # pairs = proc.filterPairs(tr_pairs)
    val = pairs[round(len(pairs)*0.95):]
    pairs = pairs[:round(len(pairs)*0.95)]

    print("TRUE")
    small_batch_size = 5
    batches = proc.batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)


    model_name = 'cb_model'
    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    loadFilename = None
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))


    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']


    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0005
    decoder_learning_ratio = 1.0
    n_iteration = 20000
    print_every = 50
    save_every = 10000

    encoder.train()
    decoder.train()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    print("Starting Training!")
    trainIters(model_name, voc, pairs, val, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)



