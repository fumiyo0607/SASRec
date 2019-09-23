import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

# args = parser.parse_args()

def get_args(dataset, train_dir, batch_size, lr, maxlen, hidden_units, num_blocks, num_epochs, num_heads, dropout_rate, l2_emb):
    args = parser.parse_args( args = [
                                        '--dataset',      dataset,
                                        '--train_dir',    train_dir,
                                        '--batch_size',   batch_size,
                                        '--lr',           lr,
                                        '--maxlen',       maxlen,
                                        '--hidden_units', hidden_units,
                                        '--num_blocks',   num_blocks,
                                        '--num_epochs',   num_epochs,
                                        '--num_heads',    num_heads,
                                        '--dropout_rate', dropout_rate,
                                        '--l2_emb',       l2_emb
                                    ]
                            )
    return args

dataset = 'Beauty'
train_dir = 'train'
batch_size = '128'
lr = '0.001'
maxlen = '50'
hidden_units = '50'
num_blocks = '2'
num_epochs = '201'
num_heads = '1'
dropout_rate = '0.5'
l2_emb = '0.0'

args = get_args(dataset, train_dir, batch_size, lr, maxlen, hidden_units, num_blocks, num_epochs, num_heads, dropout_rate, l2_emb)

def model_train(args):
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(list(vars(args).items()), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = int(len(user_train) / args.batch_size)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = Model(usernum, itemnum, args)
    sess.run(tf.initialize_all_variables())

    T = 0.0
    t0 = time.time()

    try:
        for epoch in range(1, args.num_epochs + 1):

            for step in tqdm(list(range(num_batch)), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                        model.is_training: True})

            if epoch % 20 == 0:
                t1 = time.time() - t0
                T += t1
                print('Evaluating', end=' ')
                t_test, success_users, success_users_attention_score, error_users, error_users_attention_score = evaluate(model, dataset, args, sess)
                t_valid = evaluate_valid(model, dataset, args, sess)
                print('')
                print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

                f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                f.flush()
                t0 = time.time()
    except:
        sampler.close()
        f.close()
        exit(1)

    f.close()
    sampler.close()
    print("Done")

model_train(args)