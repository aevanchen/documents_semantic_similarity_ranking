from __future__ import print_function
import sys
import random
random.seed(49999)
import numpy

sys.path.append('../matchzoo/inputs/')
sys.path.append('../matchzoo/utils/')
from preparation import Preparation
from preprocess import Preprocess, NgramUtil


if __name__ == '__main__':
    prepare = Preparation()
    srcdir = '../matchzoo/'
    dstdir = '../matchzoo/'


    ####################
    #input is quora data
    #infile = srcdir + 'quora_duplicate_questions.tsv'
    #corpus, rels = prepare.run_with_one_corpus_for_quora(infile)


    #####################
    #input is SICK data
    infile = srcdir + 'train_snli.txt'

    order=2
    #order=2， sen1,sen2, 0 or 1
    #order=1.  0 or 1, sen1, sen2
    corpus, rels = prepare.run_with_one_corpus(infile,order)
    print('total corpus : %d ...' % (len(corpus)))
    print('total relations : %d ...' % (len(rels)))
    prepare.save_corpus(dstdir + 'corpus.txt', corpus)
    prepare.save_relation(dstdir + 'relation.txt', rels)

    rel_train, rel_valid, rel_test = prepare.split_train_valid_test(rels, [0.8, 0.1, 0.1])
    prepare.save_relation(dstdir + 'relation_train.txt', rel_train)
    prepare.save_relation(dstdir + 'relation_valid.txt', rel_valid)
    prepare.save_relation(dstdir + 'relation_test.txt', rel_test)
    print('Preparation finished ...')

    #filter output stop words
    preprocessor = Preprocess(word_stem_config={'enable': False})
    #preprocessor = Preprocess(word_stem_config={'enable': False},word_filter_config={'enable':False})


    dids, docs = preprocessor.run(dstdir + 'corpus.txt')
    preprocessor.save_word_dict(dstdir + 'word_dict.txt')
    preprocessor.save_words_stats(dstdir + 'word_stats.txt')
    fout = open(dstdir + 'corpus_preprocessed.txt', 'w')
    for inum, did in enumerate(dids):
        fout.write('%s\t%s\n' % (did, ' '.join(map(str, docs[inum]))))
    fout.close()
    print('preprocess finished ...')
    print('preprocess finished ...')
