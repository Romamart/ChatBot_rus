from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from Classes import *
from Proc import Proc
from Voc import Voc
import requests
from natasha import  NamesExtractor, AddressExtractor

import os




def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
    indexes_batch = [proc.indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            if "найти" in input_sentence.lower().lstrip():
                ex = NamesExtractor()
                if ex(input_sentence) and len(ex(input_sentence)) == 1:
                    path = 'https://vk.com/search'
                    first_mask = '?c%5Bper_page%5D=40&c%5Bphoto%5D=1&c%5Bq%5D='
                    second_


            input_sentence = proc.normalizeString(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

if __name__ == '__main__':


    corpus_name = "train"
    corpus = os.path.join("Data", corpus_name)


    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    voc = Voc(corpus)

    voc.load()

    proc = Proc(10, 3)



    datafile = os.path.join(corpus, "di_all.txt")

    save_dir = os.path.join("Data", "save")

    model_name = 'cb_model'
    attn_model = 'dot'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    checkpoint_iter = 4000
    loadFilename = os.path.join(save_dir, model_name, corpus_name,
                               '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                               '{}_checkpoint.tar'.format(checkpoint_iter))



    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

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

    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    evaluateInput(encoder, decoder, searcher, voc)
